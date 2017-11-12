
import time
import random
from collections import deque
import numpy as np
import os

import gym
from gym import envs

import tensorflow as tf

class OnlineAgent:
    def __init__(self,
                 env,
                 brain,
                 max_epsilon=1.0,
                 min_epsilon=0.1,
                 epsilon_decay=0.001,
                 gamma=0.99):

        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.brain = brain
        self.epsilon = max_epsilon
        self.steps = 0
        self.steps_since_model_update = 0
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.episode_memory = []

    def episode_start(self, train):
        self.episode_memory = []

    def act(self, state):
        Q = self.brain.predictOne(state)
        action = (random.randint(0, self.env.action_size-1)
                  if random.random() < self.epsilon
                  else np.argmax(Q))
        return (action, Q[action])

    def observe(self, data, train):
        (state, action, reward_plus, next_state) = data
        self.episode_memory.append(data)
        self.steps += 1
        self.steps_since_model_update += 1

        if train:
            self.epsilon = np.maximum(self.min_epsilon, self.epsilon * (1 - self.epsilon_decay))

        if next_state is None and train:
            self._train(self.episode_memory)
            

    def _train(self, batch):

        n = len(batch)

        no_state = np.zeros(self.state_size)

        states = np.array([ o[0] for o in batch ])
        p = agent.brain.predictBatch(states, target=False)

        x = np.zeros((n, self.state_size))
        y = np.zeros((n, self.action_size))
        cum_reward = 0
        
        for i in range(n-1, -1, -1):
            (state, action, reward, next_state) = batch[i]
            target = p[i]

            cum_reward = cum_reward * self.gamma + reward
            target[action] = cum_reward

            x[i] = state
            y[i] = target

        self.brain.train(x, y)
        self.brain.updateTargetModel()
