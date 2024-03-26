import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import robosuite as suite
import yaml
from yaml.loader import SafeLoader
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config

with open('train_sac_params.yml', 'r') as file:
    train_params = list(yaml.load_all(file, Loader=SafeLoader))

for params in train_params:
    print(params["env"])

    # Environment
    # env = NormalizedActions(gym.make(args.env_name))
    config = load_controller_config(default_controller="OSC_POSE")
    env_suite = suite.make(
    env_name=params["env"], # try with other tasks like "Stack" and "Door"
    robots="Kinova3",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    controller_configs=config,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,
    )

    env = GymWrapper(env_suite)
    #env.seed(params["seed"])
    env.action_space.seed(params["seed"])

    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, params)

    #Tesnorboard
    writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), params["env"],
                                                                params["policy"], "autotune" if params["automatic_entropy_tuning"] else ""))

    # Memory
    memory = ReplayMemory(params["replay_size"], params["seed"])

    # Training Loop
    total_numsteps = 0
    updates = 0

    for i_episode in range(params['max_num_episode']):#itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        state = state[0]

        while not done:
            if  params["start_steps"] > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > params["batch_size"]:
                # Number of updates per step in environment
                for i in range(params["updates_per_step"]):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, params["batch_size"], updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1

            next_state, reward, done, _, _ = env.step(action) # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps >  params["num_steps"]:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        if i_episode % 10 == 0 and  params["eval"] is True:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                state = state[0]
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _, _ = env.step(action)
                    if 2*i_episode > params['max_num_episode']:
                        env.render()
                    episode_reward += reward


                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            writer.add_scalar('avg_reward/test', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

    agent.save_checkpoint(params['env'])
    env.close()