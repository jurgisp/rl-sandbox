
import numpy as np
from collections import deque
import threading
import random

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import tensorflow as tf

class Brain:
    _lock_tf = threading.Lock()
    _lock_mem = threading.Lock()

    def __init__(self, 
                 env,                  
                 layer1_size=64,
                 layer2_size=0,
                 opt_name='Adam',
                 opt_lr=0.001,
                 opt_loss=keras.losses.mse,
                 batch_size=32,
                 #
                 use_replay=False, # DQN vs Online
                 ddqn=False, # DQN but also applies to Online
                 target_freq=10000,
                 train_freq=32,                 
                 memory_size = 100000 # DQN only
                 ):

        self.parameters = ['layer1_size', 'layer2_size', 'opt_lr', 'batch_size',
            'use_replay', 'target_freq', 'train_freq', 'memory_size', 'ddqn']

        self.state_size = env.state_size
        self.action_size = env.action_size
        self.gamma = env.gamma

        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_loss = opt_loss
        self.batch_size = batch_size

        self.use_replay = use_replay
        self.ddqn = ddqn
        self.target_freq = target_freq
        self.train_freq  = train_freq
        self.memory_size = memory_size

        self.model = self._createModel()
        self.target_model = self._createModel()
        self.memory = self._initMemory()
        self.tf_graph = tf.get_default_graph()
        self.train_counter = 0 # How many steps since train
        self.target_counter = 0 # How many steps since target update
        self.train_nsize = 1 # Set from observation, need to know how many memories to sampl

    def get_parameters(self):
        return dict([(p, getattr(self, p)) for p in self.parameters])

    def _initMemory(self):
        return deque(maxlen=self.memory_size)

    def _createModel(self):
        model = Sequential()

        model.add(Dense(self.layer1_size, activation='relu', input_dim=self.state_size))
        if self.layer2_size > 0:
            model.add(Dense(self.layer2_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        optimizer = self._createOptimizer(self.opt_name, self.opt_lr)
        model.compile(loss=self.opt_loss, optimizer=optimizer)
        return model

    def _createOptimizer(self, opt_name, opt_lr):
        if opt_name == 'Adam':
            return Adam(opt_lr)
        if opt_name == 'RMSprop':
            return RMSprop(opt_lr)
        raise NotImplementedError("Unkonown optimizer: " + opt_name)

    def _fit(self, x, y):
        with self._lock_tf:
            with self.tf_graph.as_default(): # Hack needed when called from another thread
                batch_size = self.batch_size
                if len(y) < 2*batch_size:
                    batch_size = len(y) # Still treat it as 1 batch if slightly above
                self.model.fit(x, y, batch_size=batch_size, epochs=1, verbose=0)

    def _predictBatch(self, s, target=False):
        with self._lock_tf:
            with self.tf_graph.as_default(): # Hack needed when called from another thread
                if target:
                    return self.target_model.predict(s)
                else:
                    return self.model.predict(s)

    def _updateTargetModel(self):
        with self._lock_tf:
            with self.tf_graph.as_default():
                self.target_model.set_weights(self.model.get_weights())

    def predict(self, s, target=False):
        return self._predictBatch(s.reshape(1, self.state_size), target=target).flatten()

    def observe(self, data_sequence):
        """
        data_sequence: [(state, action, reward, next_state, Q), ...]
            Consecutive sequence of steps for n-step training
        """
        train_batch = []
        update_target = False
        self.train_nsize = len(data_sequence)

        with self._lock_mem:
            # Lock while updating memory and counters (should be fast)
            self.memory.append(data_sequence)
            self.train_counter += len(data_sequence)
            self.target_counter += len(data_sequence)

            # Time to train?
            if self.train_counter >= self.train_freq:
                self.train_counter = 0
                if self.use_replay:
                    # DQN version - sample experience
                    sample_size = self.batch_size // self.train_nsize
                    train_batch = random.sample(self.memory, np.minimum(len(self.memory), sample_size))
                else:
                    # Online version - take all experience
                    train_batch = list(self.memory)
                    self.memory = self._initMemory()

            # Time to update target?
            if self.target_counter >= self.target_freq:
                self.target_counter = 0
                update_target = True

        # Train
        if len(train_batch) > 0:
            self._train(train_batch)

        # Target update
        if update_target:
            self._updateTargetModel()

    def _train(self, batch_sequences):
        """
        batch_sequences:
            List of sequences of consecutive N (or less) steps
            [
                [(state, action, reward, next_state, Q), ..., (step n)]
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

        no_state = np.zeros(self.state_size)
        states = np.array([ o[0] for o in batch ])
        next_states = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        states_q = np.array([ o[4] for o in batch ])
        if self.use_replay:
            # If replaying from experience, re-predict Q, otherwise take as it was
            states_q = self._predictBatch(states, target=False)
        
        next_states_q = None
        if self.ddqn:
            # For double-DQN we also need 
            next_states_q = self._predictBatch(next_states, target=False)

        next_states_qtarget = self._predictBatch(next_states, target=True)

        # Build x and y

        x = np.zeros((n, self.state_size))
        y = np.zeros((n, self.action_size))
        cum_reward = 0

        for i in range(n-1, -1, -1):
            (state, action, reward, next_state, _) = batch[i]

            if is_sequence_end[i] or action != np.argmax(states_q[i]):
                # Need to reinitialize target reward from prediction if:
                #   1) End of sequence
                #   2) The action taken wasn't according to Q (because of epsilon, or experience replay)
                if next_state is None:
                    cum_reward = 0
                else:
                    qtarget = next_states_qtarget[i]
                    if self.ddqn:
                        cum_reward = qtarget[ np.argmax(next_states_q[i]) ]
                    else:
                        cum_reward = np.max(qtarget)
            else:
                # DEBUG
                if i < n-1 and not np.array_equal(next_states[i], states[i+1]):
                    raise Exception('Expecting consecutive states')

            cum_reward = cum_reward * self.gamma + reward
            target = np.copy(states_q[i])
            target[action] = cum_reward

            x[i] = state
            y[i] = target

        self._fit(x, y)
