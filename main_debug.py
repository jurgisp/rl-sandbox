from agent import Agent
from brain import Brain
from environment import Environment

import gym
from gym import envs

if __name__ == '__main__':

    envs.register(
        id='MountainCar-v1',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        tags={'wrapper_config.TimeLimit.max_episode_steps': 190},
        reward_threshold=190,
    )
    
    env = Environment('MountainCar-v1', run_name=None, repeat_steps=5, gamma=0.99)

    brain = Brain(env, layer1_size = 32, layer2_size = 32, opt_name = 'Adam', opt_lr = 1e-3,
                    target_freq = 10000, train_freq = 32, use_replay = True, ddqn = True)

    agent = Agent(env, brain, min_epsilon = 0.1, epsilon_decay = 1/5e5, train_nsteps = 4)

    env.run(agent, train=True, explore=True)
    env.run(agent, train=True, explore=True)
    env.run(agent, train=True, explore=True)

