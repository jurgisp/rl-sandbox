
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
                 # Common
                 target_freq=1000,
                 train_freq=5,
                 train_nsteps=5,
                 max_epsilon=1.0,
                 min_epsilon=0.1,
                 epsilon_decay=0.001,
                 gamma=0.99,
                 # Online
                 sarsa=False):


        self.parameters = ['target_freq', 'train_freq', 'train_nsteps', 'epsilon_decay', 'gamma', 'sarsa']
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
        self.train_nsteps = train_nsteps

        self.memory = []
        self.sarsa = sarsa

    def episode_start(self, train):
        return

    def act(self, state, explore=True):
        Q = self.brain.predictOne(state)
        action = (random.randint(0, self.action_size-1)
                  if explore and random.random() < self.epsilon
                  else np.argmax(Q))
        return (action, Q)

    def observe(self, data, train, global_step):
        (state, action, reward, next_state, Q, next_action) = data
        self.steps += 1
        self.epsilon = np.maximum(self.min_epsilon, self.max_epsilon * np.exp(-global_step * self.epsilon_decay))

        if train:
            self.memory.append(data)
            
            if self.steps % self.train_freq == 0:
                self._train(self.memory)
                self.memory = []

            if global_step % self.target_freq == 0:
                self.brain.updateTargetModel()
            

    def _train(self, batch):

        n = len(batch)
        x = np.zeros((n, self.state_size))
        y = np.zeros((n, self.action_size))

        cum_reward = 0
        last_state = np.zeros(self.state_size)
        last_Q = np.zeros(self.action_size)
        n_steps = 0

        for i in range(n-1, -1, -1):
            (state, action, reward, next_state, Q, next_action) = batch[i]
            continuous = np.array_equal(next_state, last_state) # Indicates if samples come from continued episode

            if not continuous or n_steps == self.train_nsteps:
                # Need to reinitialize target reward from prediction if:
                #   1) Discontinuity in states (i.e. batch[i+1].state != batch[i].next_state)
                #   2) Number of train_nsteps reached
                n_steps = 0
                if next_state is None:
                    cum_reward = 0
                else:
                    next_Q = self.brain.predictOne(next_state, target=True)
                    cum_reward = (next_Q[next_action]
                        if self.sarsa else
                        np.max(next_Q)
                    )

            cum_reward = cum_reward * self.gamma + reward
            target = np.copy(Q)
            target[action] = cum_reward

            x[i] = state
            y[i] = target

            last_state = state
            last_Q = Q
            n_steps += 1

        self.brain.train(x, y)

    def get_parameters(self):
        agent_parameters = dict([(p, getattr(self, p)) for p in self.parameters])
        brain_parameters = self.brain.get_parameters()
        return {**agent_parameters, **brain_parameters}
