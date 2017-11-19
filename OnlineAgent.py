
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
                 gamma=0.99,
                 sarsa=False):

        self.parameters = ['target_freq', 'train_freq', 'epsilon_decay', 'gamma', 'sarsa']
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.brain = brain
        self.max_epsilon = max_epsilon
        self.epsilon = max_epsilon
        self.steps = 0
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_freq = target_freq
        self.train_freq  = train_freq
        self.memory = []
        self.sarsa = sarsa

    def episode_start(self, train):
        self.memory = []

    def act(self, state):
        Q = self.brain.predictOne(state)
        action = (random.randint(0, self.action_size-1)
                  if random.random() < self.epsilon
                  else np.argmax(Q))
        return (action, Q)

    def observe(self, data, train, global_step):
        (state, action, reward, next_state, Q, next_action) = data
        
        self.steps += 1
        self.epsilon = np.maximum(self.min_epsilon, self.max_epsilon * np.exp(-global_step * self.epsilon_decay))

        if train:
            self.memory.append(data)
            
            if self.steps % self.train_freq == 0 or next_state is None:
                self._train(self.memory)
                self.memory = []

            if global_step % self.target_freq == 0:
                self.brain.updateTargetModel()
            

    def _train(self, batch):

        n = len(batch)
        x = np.zeros((n, self.state_size))
        y = np.zeros((n, self.action_size))

        cum_reward = 0
        (state, action, reward, next_state, Q, next_action) = batch[-1]
        if next_state is not None:
            Q = self.brain.predictOne(next_state, target=True)
            if self.sarsa:
                cum_reward = Q[next_action]
            else:
                cum_reward = np.max(Q)
                
        
        for i in range(n-1, -1, -1):
            (state, action, reward, next_state, Q, next_action) = batch[i]
            target = Q

            cum_reward = cum_reward * self.gamma + reward
            target[action] = cum_reward

            x[i] = state
            y[i] = target

        self.brain.train(x, y)

    def get_parameters(self):
        agent_parameters = dict([(p, getattr(self, p)) for p in self.parameters])
        brain_parameters = self.brain.get_parameters()
        return {**agent_parameters, **brain_parameters}
