
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
                 memory_size = 1000000,
                 max_epsilon = 1.0,
                 min_epsilon = 0.1,
                 epsilon_decay = 0.001,
                 batch_size = 64,
                 gamma = 0.99,
                 target_freq = 1000,
                 use_target = False):

        self.env = env
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.brain = brain
        self.memory_size = memory_size
        self.memory = self.Memory(memory_size)
        self.epsilon = max_epsilon
        self.steps = 0
        self.steps_since_model_update = 0
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_freq = target_freq
        self.use_target = use_target

    def episode_start(self, train):
        if train:
            self._try_update_target_model()

    def act(self, state):
        Q = self.brain.predictOne(state, target=self.use_target)
        action = (random.randint(0, self.env.action_size-1)
                  if random.random() < self.epsilon
                  else np.argmax(Q))
        return (action, Q[action], Q)

    def observe(self, data, train):
        # (state, action, reward, next_state, Q, next_action) = data
        self.memory.add(data)
        self.steps += 1
        if train:
            self._replay()

    def _replay(self):

        batch = self.memory.sample(self.batch_size)
        n = len(batch)

        no_state = np.zeros(self.state_size)

        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        
        p = agent.brain.predictBatch(states, target=False)
        p_ = agent.brain.predictBatch(states_, target=False)
        pTarget_ = agent.brain.predictBatch(states_, target=True)

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
        
        self.epsilon = np.maximum(self.min_epsilon, self.epsilon * (1 - self.epsilon_decay))
        self.steps_since_model_update += 1

            
    def _try_update_target_model(self):
        if self.steps_since_model_update >= self.target_freq:
            self.steps_since_model_update = 0
            self.brain.updateTargetModel()

    class Memory:
        def __init__(self, capacity):
            self.samples = deque(maxlen=capacity)

        def add(self, sample):
            self.samples.append(sample)

        def sample(self, n):
            n = min(n, len(self.samples))
            return random.sample(self.samples, n)