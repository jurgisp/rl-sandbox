
import numpy as np
from collections import deque
import threading
import random

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

class BrainAC:
    _lock_tf = threading.Lock()
    _lock_mem = threading.Lock()

    def __init__(self, 
                 env,                  
                 layer1_size=64,
                 layer2_size=0,
                 opt_name='Adam',
                 opt_lr=0.001,
                 batch_size=32,
                 #
                 use_replay=False, # DQN vs Online
                 train_freq=32,                 
                 memory_size = 100000 # DQN only
                 ):

        self.parameters = ['layer1_size', 'layer2_size', 'opt_lr', 'batch_size',
            'use_replay', 'train_freq', 'memory_size']

        self.state_size = env.state_size
        self.action_size = env.action_size
        self.gamma = env.gamma

        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.batch_size = batch_size

        self.use_replay = use_replay
        self.train_freq  = train_freq
        self.memory_size = memory_size

        self.policy_model = self._createPolicyModel()
        self.value_model = self._createValueModel()
        self.memory = self._initMemory()
        self.tf_graph = tf.get_default_graph()
        self.train_counter = 0 # How many steps since train
        self.train_nsize = 1 # Set from observation, need to know how many memories to sampl

    def get_parameters(self):
        return dict([(p, getattr(self, p)) for p in self.parameters])

    def _initMemory(self):
        return deque(maxlen=self.memory_size)

    def _createPolicyModel(self):
        model = Sequential()

        model.add(Dense(self.layer1_size, activation='relu', input_dim=self.state_size))
        if self.layer2_size > 0:
            model.add(Dense(self.layer2_size, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))

        optimizer = self._createOptimizer(self.opt_name, self.opt_lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def _createValueModel(self):
        model = Sequential()

        model.add(Dense(self.layer1_size, activation='relu', input_dim=self.state_size))
        if self.layer2_size > 0:
            model.add(Dense(self.layer2_size, activation='relu'))
        model.add(Dense(1, activation='linear'))

        optimizer = self._createOptimizer(self.opt_name, self.opt_lr)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def _createOptimizer(self, opt_name, opt_lr):
        if opt_name == 'Adam':
            return Adam(opt_lr)
        if opt_name == 'RMSprop':
            return RMSprop(opt_lr)
        raise NotImplementedError("Unkonown optimizer: " + opt_name)

    def _fit(self, x, y_value, y_advantage):
        with self._lock_tf:
            with self.tf_graph.as_default(): # Hack needed when called from another thread
                batch_size = self.batch_size
                if len(y_value) < 2*batch_size:
                    batch_size = len(y_value) # Still treat it as 1 batch if slightly above

                self.value_model.fit(x, y_value, batch_size=batch_size, epochs=1, verbose=0)
                self.policy_model.fit(x, y_advantage, batch_size=batch_size, epochs=1, verbose=0)

    def _predict_action_batch(self, s):
        with self._lock_tf:
            with self.tf_graph.as_default(): # Hack needed when called from another thread
                return self.policy_model.predict(s)

    def predict_action(self, s):
        return self._predict_action_batch(s.reshape(1, self.state_size)).flatten()

    def _predict_value_batch(self, s):
        with self._lock_tf:
            with self.tf_graph.as_default(): # Hack needed when called from another thread
                return self.value_model.predict(s)

    def predict_value(self, s):
        return self._predict_value_batch(s.reshape(1, self.state_size)).flatten()[0]

    def observe(self, data_sequence):
        """
        data_sequence: [[state, action, reward, next_state, p_act, v, v_next], ...]
            Consecutive sequence of steps for n-step training
        """
        train_batch = []
        self.train_nsize = len(data_sequence)

        with self._lock_mem:
            # Lock while updating memory and counters (should be fast)
            self.memory.append(data_sequence)
            self.train_counter += len(data_sequence)

            # Time to train?
            if self.train_counter >= self.train_freq:
                self.train_counter = 0
                if self.use_replay:
                    # DQN version - sample experience
                    raise Exception('Not implemented')
                    # sample_size = self.batch_size // self.train_nsize
                    # train_batch = random.sample(self.memory, np.minimum(len(self.memory), sample_size))
                else:
                    # Online version - take all experience
                    train_batch = list(self.memory)
                    self.memory = self._initMemory()

        # Train
        if len(train_batch) > 0:
            self._train(train_batch)

    def _train(self, batch_sequences):
        """
        batch_sequences:
            List of sequences of consecutive N (or less) steps
            [
                [[state, action, reward, next_state, p_act, v, v_next], ..., (step n)]
                [... sequence 2 ...]
                ...
                [... sequence M ...]
            ]
        """
        batch = np.concatenate(batch_sequences)
        n = len(batch)

        # Mark where are endpoints of sequences
        is_sequence_end = np.array([False for _ in range(n)])
        i = 0
        for seq in batch_sequences:
            i += len(seq)
            is_sequence_end[i-1] = True

        # Build x and y

        x = np.zeros((n, self.state_size))
        y_value = np.zeros((n, 1))
        y_advantage = np.zeros((n, self.action_size))
        cum_reward = 0

        for i in range(n-1, -1, -1):
            [state, action, reward, next_state, p_act, v, v_next] = batch[i]

            if is_sequence_end[i]:
                # Need to reinitialize target reward from prediction
                cum_reward = v_next

            cum_reward = cum_reward * self.gamma + reward

            x[i] = state
            y_value[i][0] = cum_reward
            y_advantage[i][action] = cum_reward - v

        self._fit(x, y_value, y_advantage)
