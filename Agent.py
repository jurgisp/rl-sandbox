
import time
import random
from collections import deque
import numpy as np
import os

import gym
from gym import envs

class Agent:
    def __init__(self,
                 env,
                 brain,
                 memory_size=1,
                 max_epsilon=1.0,
                 min_epsilon=0.1,
                 epsilon_decay=0.001):


        self.parameters = ['epsilon_decay']
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.brain = brain
        self.memory_size = memory_size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.epsilon = max_epsilon
        self.memory = []

    def get_parameters(self):
        agent_parameters = dict([(p, getattr(self, p)) for p in self.parameters])
        brain_parameters = self.brain.get_parameters()
        return {**agent_parameters, **brain_parameters}

    def get_epsilon(self, step):
        return np.maximum(self.min_epsilon, self.max_epsilon * np.exp(-step * self.epsilon_decay))

    def episode_start(self, train):
        return

    def act(self, state, global_step, explore):
        Q = self.brain.predict(state)
        action = (random.randint(0, self.action_size-1)
                  if explore and random.random() < self.get_epsilon(global_step)
                  else np.argmax(Q))
        return (action, Q)

    def observe(self, data, train):
        if train:
            self.memory.append(data)
            if len(self.memory) >= self.memory_size:
                data_batch = self.memory
                self.memory = []
                self.brain.observe(data_batch)


