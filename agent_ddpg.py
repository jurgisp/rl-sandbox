
import time
import random
from collections import deque
import numpy as np
import os

import gym
from gym import envs
from .noise import OrnsteinUhlenbeckActionNoise

class AgentDDPG:
    def __init__(self,
                 env,
                 brain,
                 train_nsteps=1,
                 noise_sigma=0.2,
                 noise_dt=1e-2):

        self.parameters = ['noise_sigma', 'train_nsteps']
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.brain = brain
        self.train_nsteps = train_nsteps
        self.noise_sigma = noise_sigma
        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(self.action_size), 
            sigma=float(self.noise_sigma) * np.ones(self.action_size),
            dt=noise_dt)

        self.memory = [] # Keeps short experience sequence of train_nsteps

    def get_parameters(self):
        agent_parameters = dict([(p, getattr(self, p)) for p in self.parameters])
        brain_parameters = self.brain.get_parameters()
        return {**agent_parameters, **brain_parameters}

    def _get_noise(self, step):
        return self.noise()

    def episode_start(self, train):
        self.memory = []
        self.noise.reset()
        return

    def act(self, state, global_step, explore):
        action = self.brain.predict_action(state)
        v = self.brain.predict_value(state)
        noise_mag = 0.
        if explore:
            noise = self._get_noise(global_step)
            noise_mag = np.sqrt(np.sum(noise * noise))
            action += noise
        action = np.clip(action, -1., 1.)
        return [action, v, noise_mag, None]

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


