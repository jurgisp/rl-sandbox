
import numpy as np
from collections import deque
import threading
import random

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras.backend.common import floatx, epsilon
import tensorflow as tf

class BrainACQ:
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
                 target_freq=32,
                 train_freq=32,                 
                 memory_size = 100000, # DQN only
                 entropy_beta=0.1
                 ):

        self.parameters = ['layer1_size', 'layer2_size', 'opt_lr', 'batch_size',
            'use_replay', 'target_freq', 'train_freq', 'memory_size', 'entropy_beta']

        self.state_size = env.state_size
        self.action_size = env.action_size
        self.gamma = env.gamma

        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.batch_size = batch_size

        self.use_replay = use_replay
        self.target_freq = target_freq
        self.train_freq  = train_freq
        self.memory_size = memory_size
        self.entropy_beta = entropy_beta

        self.policy_model = self._createPolicyModel(self.entropy_beta)
        self.policy_model_target = self._createPolicyModel(self.entropy_beta)
        self.value_model = self._createValueModel()
        self.value_model_target = self._createValueModel()
        self.memory = self._initMemory()
        self.tf_graph = tf.get_default_graph()
        self.train_counter = 0 # How many steps since train
        self.target_counter = 0 # How many steps since target update
        self.train_nsize = 1 # Set from observation, need to know how many memories to sampl

    def get_parameters(self):
        return dict([(p, getattr(self, p)) for p in self.parameters])

    def _initMemory(self):
        return deque(maxlen=self.memory_size)

    def _createPolicyModel(self, entropy_beta):

        def categorical_crossentropy_regularized(target, output):
            """Categorical crossentropy between an output tensor and a target tensor.
            # Arguments
                target: A tensor of the same shape as `output`.
                output: A tensor resulting from a softmax
                    (unless `from_logits` is True, in which
                    case `output` is expected to be the logits).
            # Returns
                Output tensor.
            """
            # scale preds so that the class probas of each sample sum to 1
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)
            # manual computation of crossentropy
            _epsilon = tf.convert_to_tensor(epsilon(), dtype=output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            return - tf.reduce_sum(target * tf.log(output)
                                - entropy_beta * output * tf.log(output),
                                    len(output.get_shape()) - 1)

        model = Sequential()

        model.add(Dense(self.layer1_size, activation='relu', input_dim=self.state_size))
        if self.layer2_size > 0:
            model.add(Dense(self.layer2_size, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))

        optimizer = self._createOptimizer(self.opt_name, self.opt_lr)
        model.compile(loss=categorical_crossentropy_regularized, optimizer=optimizer)
        return model

    def _createValueModel(self):
        model = Sequential()

        model.add(Dense(self.layer1_size, activation='relu', input_dim=self.state_size))
        if self.layer2_size > 0:
            model.add(Dense(self.layer2_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

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

    def _predict_action_batch(self, s, target=False):
        with self._lock_tf:
            with self.tf_graph.as_default(): # Hack needed when called from another thread
                if target:
                    return self.policy_model_target.predict(s)
                else:
                    return self.policy_model.predict(s)

    def predict_action(self, s):
        return self._predict_action_batch(s.reshape(1, self.state_size)).flatten()

    def _predict_value_batch(self, s, target=False):
        with self._lock_tf:
            with self.tf_graph.as_default(): # Hack needed when called from another thread
                if target:
                    return self.value_model_target.predict(s)
                else:
                    return self.value_model.predict(s)

    def predict_value(self, s):
        return self._predict_value_batch(s.reshape(1, self.state_size)).flatten()

    def _updateTargetModel(self):
        with self._lock_tf:
            with self.tf_graph.as_default():
                self.policy_model_target.set_weights(self.policy_model.get_weights())
                self.value_model_target.set_weights(self.value_model.get_weights())

    def observe(self, data_sequence):
        """
        data_sequence: [[state, action, reward, next_state, [q, p_act]], ...]
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
                [[state, action, reward, next_state, [q, p_act]], ..., (step n)]
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

        states_q = np.array([ o[4][0] for o in batch ])
        states_p = np.array([ o[4][1] for o in batch ])
        if self.use_replay:
            # If replaying from experience, re-predict Q, otherwise take as it was
            states_q = self._predict_value_batch(states, target=False)
            states_p = self._predict_action_batch(states, target=False)
        
        next_states_q = self._predict_value_batch(next_states, target=True)
        next_states_p = self._predict_action_batch(next_states, target=True)

        # Build x and y

        x = np.zeros((n, self.state_size))
        y_value = np.zeros((n, self.action_size))
        y_advantage = np.zeros((n, self.action_size))
        cum_reward = 0

        for i in range(n-1, -1, -1):
            [state, action, reward, next_state, _] = batch[i]

            if is_sequence_end[i]:
                # Need to reinitialize target reward from prediction
                if next_state is None:
                    cum_reward = 0
                else:
                    qtarget = next_states_q[i]
                    ptarget = next_states_p[i]
                    #cum_reward = np.sum(qtarget * ptarget)
                    cum_reward = qtarget[np.argmax(ptarget)]

            cum_reward = cum_reward * self.gamma + reward

            if random.random() < 0.01:
                print(action, states_q[i], states_p[i])

            x[i] = state
            y_value[i] = np.copy(states_q[i])
            y_value[i][action] = cum_reward
            y_advantage[i] = np.copy(states_q[i]) # Not actually advantage, just reward

        self._fit(x, y_value, y_advantage)
