
import time
import random
from collections import deque
import numpy as np
import os

import gym
from gym import envs

class AgentAC:
    def __init__(self,
                 env,
                 brain,
                 train_nsteps=1,
                 max_epsilon=0.0,
                 min_epsilon=0.0,
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
        p_action = self.brain.predict_action(state)
        v = self.brain.predict_value(state)
        action = np.random.choice(self.action_size, p=p_action)
        if random.random() < self._get_epsilon(global_step):
            action = random.randint(0, self.action_size-1)
        return [action, v, 1 - p_action[action], [v, p_action]]

    def observe(self, data, train, done):
        """
        data: [state, action, reward, next_state, [v, p_act]
        """
        if train:
            if len(self.memory) > 0:
                if self.memory[-1][3] is None or not np.array_equal(self.memory[-1][3], data[0]):
                    raise Exception('Expecting consecutive states')

                self.memory[-1][6] = data[4][0] # Fill v_next

            # Experience sequence terminates when reaches (train_nsteps)
            if len(self.memory) >= self.train_nsteps:
                data_sequence = self.memory
                self.memory = []
                self.brain.observe(data_sequence)

            # (state, action, reward, next_state, p_act, v, v_next)
            self.memory.append([data[0], data[1], data[2], data[3], data[4][1], data[4][0], None])

            if done and (data[3] is None):
                # Submit ending sequence only on actual game-over (next_state is None), not on time-limit
                self.memory[-1][6] = 0.
                self.brain.observe(self.memory)


