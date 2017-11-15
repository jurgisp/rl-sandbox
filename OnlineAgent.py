
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
                 target_freq=1000,
                 train_freq=5,
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
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_freq = target_freq
        self.train_freq  = train_freq
        self.memory = []

    def episode_start(self, train):
        self.memory = []

    def act(self, state):
        Q = self.brain.predictOne(state)
        action = (random.randint(0, self.env.action_size-1)
                  if random.random() < self.epsilon
                  else np.argmax(Q))
        return (action, Q[action], Q)

    def observe(self, data, train):
        (state, action, reward, next_state, Q) = data
        self.memory.append(data)
        self.steps += 1

        if train:
            if self.steps % self.train_freq == 0 or next_state is None:
                self._train(self.memory)
                self.memory = []

            if self.steps % self.target_freq == 0:
                self.brain.updateTargetModel()

            self.epsilon = np.maximum(self.min_epsilon, self.epsilon * (1 - self.epsilon_decay))
            

    def _train(self, batch):

        n = len(batch)
        x = np.zeros((n, self.state_size))
        y = np.zeros((n, self.action_size))

        cum_reward = 0
        last_state = batch[-1][3]
        if last_state is not None:
            Q = self.brain.predictOne(last_state, target=True)
            cum_reward = np.max(Q)
        
        for i in range(n-1, -1, -1):
            (state, action, reward, next_state, Q) = batch[i]
            target = Q

            cum_reward = cum_reward * self.gamma + reward
            target[action] = cum_reward

            x[i] = state
            y[i] = target

        self.brain.train(x, y)
