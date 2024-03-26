import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque
import robosuite as suite
import yaml
from yaml.loader import SafeLoader
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 

from utils.utils import *
from utils.zfilter import ZFilter
from model import Actor, Critic, Discriminator
import h5py
from train_model import train_actor_critic, train_discrim

parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--env_name', type=str, default="Hopper-v2", 
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None, 
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False, 
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=100, 
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--discrim_update_num', type=int, default=2, 
                    help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=2048, 
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=4000,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


def main():
    with open('train_sac_params.yml', 'r') as file:
        train_params = list(yaml.load_all(file, Loader=SafeLoader))
    params = train_params[0]
    config = load_controller_config(default_controller="OSC_POSE")
    env_suite = suite.make(
        env_name=params["env"],  # try with other tasks like "Stack" and "Door"
        robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        controller_configs=config,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )

    env = GymWrapper(env_suite)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0] - 10
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)

    print('state size:', num_inputs) 
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)
    discrim = Discriminator(num_inputs + num_actions, args)

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate, 
                              weight_decay=args.l2_rate) 
    discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)
    
    # load demonstrations
    f = h5py.File("expert_demo/Lift/demo.hdf5", "r")
    demos = list(f["data"].keys())
    states = np.array(f["data/{}/states".format(demos[0])][()])
    actions = np.array(f["data/{}/actions".format(demos[0])][()])
    d = np.hstack([states,actions])
    for ep in demos[1:]:
        states = np.array(f["data/{}/states".format(ep)][()])
        actions = np.array(f["data/{}/actions".format(ep)][()])
        de = np.hstack([states,actions])
        d = np.vstack([d,de])
        # expert_demo, _ = pickle.load(open('./expert_demo/expert_demo.p', "rb")) # demonstrations = np.array(expert_demo)
    demonstrations = d
    # print("demonstrations.shape", demonstrations.shape)
    print("demonstrations.shape", demonstrations.shape)
    
    writer = SummaryWriter(args.logdir)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        discrim.load_state_dict(ckpt['discrim'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    
    episodes = 0
    train_discrim_flag = True

    for iter in range(args.max_iter_num):
        actor.eval(), critic.eval()
        memory = deque()
        reward_memory = deque()

        steps = 0
        scores = []

        while steps < args.total_sample_size: 
            state,_ = env.reset()
            score = 0
            print(state.shape)
            state = state[:40]
            state = running_state(state)
            
            for _ in range(10000): 
                if args.render:
                    env.render()

                steps += 1

                mu, std = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state[:40]
                irl_reward = get_reward(discrim, state, action)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append(np.hstack([state, action]))
                reward_memory.append([irl_reward, mask])
                next_state = running_state(next_state)
                state = next_state

                score += reward

                if done:
                    break
            
            episodes += 1
            scores.append(score)
        
        score_avg = np.mean(scores)
        print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), iter)

        actor.train(), critic.train(), discrim.train()
        if train_discrim_flag:
            expert_acc, learner_acc = train_discrim(discrim, memory, reward_memory, discrim_optim, demonstrations, args)
            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
            if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                train_discrim_flag = False
        train_actor_critic(actor, critic, memory, reward_memory, actor_optim, critic_optim, args)

        if iter % 100:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(),'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_'+ str(score_avg)+'.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'discrim': discrim.state_dict(),
                'z_filter_n':running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)

if __name__=="__main__":
    main()