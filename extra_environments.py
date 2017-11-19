
import gym
from gym import envs

if 'CartPole-v1000' not in envs.registry.env_specs:
    envs.register(
        id='CartPole-v1000',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
        reward_threshold=1000,
    )

if 'MountainCar-v1000' not in envs.registry.env_specs:
    envs.register(
        id='MountainCar-v1000',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
        reward_threshold=1000,
    )

if 'MountainCar-v10000' not in envs.registry.env_specs:
    envs.register(
        id='MountainCar-v10000',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
        reward_threshold=10000,
    )

if 'MountainCar-v100000' not in envs.registry.env_specs:
    envs.register(
        id='MountainCar-v100000',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
        reward_threshold=100000,
    )