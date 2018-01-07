
import time
import random
from collections import deque
import numpy as np
import os

import gym
from gym import envs

import tensorflow as tf

class DqnAgent:
    def __init__(self, 
                 env, 
                 brain,
                 # Common
                 target_freq=1000,
                 train_freq=5,
                 max_epsilon=1.0,
                 min_epsilon=0.1,
                 epsilon_decay=0.001,
                 gamma=0.99,
                 # DQN
                 memory_size=100000,
                 batch_size=64):

        self.parameters = ['target_freq', 'train_freq', 'epsilon_decay', 'gamma', 'memory_size', 'batch_size']
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

        self.memory_size = memory_size
        self.memory = self.Memory(memory_size)
        self.batch_size = batch_size

    def episode_start(self, train):
        return

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

        self.memory.add(data)

        if train:
            if self.steps % self.train_freq == 0:
                batch = self.memory.sample(self.batch_size)
                self._train(batch)

            if global_step % self.target_freq == 0:
                self.brain.updateTargetModel()

    def _train(self, batch):

        n = len(batch)

        no_state = np.zeros(self.state_size)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        
        p = self.brain.predictBatch(states, target=False)
        p_ = self.brain.predictBatch(states_, target=False)
        pTarget_ = self.brain.predictBatch(states_, target=True)

        x = np.zeros((n, self.state_size))
        y = np.zeros((n, self.action_size))
        
        for i in range(n):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * pTarget_[i][ np.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

    def get_parameters(self):
        agent_parameters = dict([(p, getattr(self, p)) for p in self.parameters])
        brain_parameters = self.brain.get_parameters()
        return {**agent_parameters, **brain_parameters}

    class Memory:
        def __init__(self, capacity):
            self.samples = deque(maxlen=capacity)
            self._next_batch = []

        def add(self, sample):
            self.samples.append(sample)
            self._next_batch.append(self.samples[random.randint(0, len(self.samples)-1)])

        def sample(self, n):
            res = self._next_batch
            self._next_batch = []
            return res
