
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
                 train_nsteps=1,
                 max_epsilon=1.0,
                 min_epsilon=0.1,
                 epsilon_decay=0.001):

        self.parameters = ['epsilon_decay', 'train_nsteps']
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.brain = brain
        self.train_nsteps = train_nsteps
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        self.memory = [] # Keeps short experience sequence of train_nsteps

    def get_parameters(self):
        agent_parameters = dict([(p, getattr(self, p)) for p in self.parameters])
        brain_parameters = self.brain.get_parameters()
        return {**agent_parameters, **brain_parameters}

    def _get_epsilon(self, step):
        return np.maximum(self.min_epsilon, self.max_epsilon * np.exp(-step * self.epsilon_decay))

    def episode_start(self, train):
        self.memory = []
        return

    def act(self, state, global_step, explore):
        q = self.brain.predict(state)
        eps = self._get_epsilon(global_step)
        action = np.argmax(q)
        if explore and random.random() < eps:
            action = random.randint(0, self.action_size-1)
        return [action, q[action], eps, q]

    def observe(self, data, train, done):
        """
        data: [state, action, reward, next_state, q]
        """
        if train:
            self.memory.append(data)
            if len(self.memory) >= self.train_nsteps or done:
                data_sequence = self.memory
                self.memory = []
                self.brain.observe(data_sequence)


