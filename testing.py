import gym_xarm
import gym

config = {
        'GUI': True,
        'init_grasp_rate': 0.0,
        'goal_ground_rate': 0.0,
        'num_obj': 1,
        'reward_type': 'incremental',
        'goal_shape': 'ground' # air, tower
    }
env = gym.make('XarmRearrange-v1', config = config)
env.reset()
for _ in range(1000):
    env.render()
    act = env.action_space.sample()
    obs, reward, done, info = env.step(act)
env.close()