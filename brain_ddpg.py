
import numpy as np
from collections import deque
import threading
import random

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from keras.backend.common import floatx, epsilon
import tensorflow as tf

class BrainDDPG:
    _lock_tf = threading.Lock()
    _lock_mem = threading.Lock()

    def __init__(self, 
                 env,                  
                 layer1_size=64,
                 layer2_size=64,
                 opt_critic='Adam',
                 opt_critic_lr=1e-3,
                 opt_actor='Adam',
                 opt_actor_lr=1e-3,
                 batch_size=32,
                 #
                 use_replay=True,
                 target_tau=1e-3,
                 train_freq=32,                 
                 memory_size=100000):

        self.parameters = ['layer1_size', 'layer2_size', 'opt_critic_lr', 'opt_actor_lr', 'batch_size',
            'use_replay', 'target_tau', 'train_freq', 'memory_size']

        self.state_size = env.state_size
        self.action_size = env.action_size
        self.gamma = env.gamma

        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.opt_critic = opt_critic
        self.opt_critic_lr = opt_critic_lr
        self.opt_actor = opt_actor
        self.opt_actor_lr = opt_actor_lr
        self.batch_size = batch_size

        self.use_replay = use_replay
        self.target_tau = target_tau
        self.train_freq  = train_freq
        self.memory_size = memory_size

        (self.actor_model, self.critic_model, self.actor_critic_model) = self._createModel()
        (_, _, self.actor_critic_target) = self._createModel()
        self.tf_graph = tf.get_default_graph()
        self._updateTargetModel(1.0)
        self.memory = self._initMemory()
        self.train_counter = 0 # How many steps since train
        self.target_counter = 0 # How many steps since target update
        self.train_nsize = 1 # Set from observation, need to know how many memories to sampl

    def get_parameters(self):
        return dict([(p, getattr(self, p)) for p in self.parameters])

    def _initMemory(self):
        return deque(maxlen=self.memory_size)

    def _createModel(self):
        """Create actor and critic models
        
        Returns:
            (actor, critic, actor_critic)
        """
        def minimize(y_true, y_pred):
            return keras.backend.mean(y_pred, axis=-1)

        actor_layer1 = Dense(self.layer1_size, activation='relu')
        actor_layer2 = Dense(self.layer2_size, activation='relu')
        actor_layer3 = Dense(self.action_size, activation='tanh')

        critic_layer1 = Dense(self.layer1_size, activation='relu')
        critic_layer2 = Dense(self.layer2_size, activation='relu')
        critic_layer3 = Dense(1, activation='linear')

        # Actor model - only for action, not training
        actor_input_state = Input(shape=(self.state_size,))
        actor_output = actor_layer3(
            actor_layer2(
                actor_layer1(actor_input_state)))
        actor_model = Model([actor_input_state], [actor_output])

        # Critic model - for training
        critic_input_state = Input(shape=(self.state_size,))
        critic_input_action = Input(shape=(self.action_size,))
        critic_output = critic_layer3(
            critic_layer2(
                keras.layers.concatenate([
                    critic_layer1(critic_input_state),
                    critic_input_action])))
        critic_model = Model([critic_input_state, critic_input_action], [critic_output])
        critic_model.compile(loss='mse',
                             optimizer=self._optimizer(self.opt_critic, self.opt_critic_lr))

        # Actor-critic model - for training the actor part, minimizing the critic
        critic_layer1.trainable = False
        critic_layer2.trainable = False
        actor_input_state = Input(shape=(self.state_size,))
        actor_output = actor_layer3(
            actor_layer2(
                actor_layer1(actor_input_state)))
        critic_input_state = actor_input_state
        critic_input_action = actor_output # This is where the models are connected
        critic_output = critic_layer3(
            critic_layer2(
                keras.layers.concatenate([
                    critic_layer1(critic_input_state),
                    critic_input_action])))
        actor_critic_model = Model([actor_input_state], [critic_output])
        actor_critic_model.compile(loss=minimize,
                                   optimizer=self._optimizer(self.opt_actor, self.opt_actor_lr))

        return (actor_model, critic_model, actor_critic_model)

    def _optimizer(self, opt_name, opt_lr):
        if opt_name == 'Adam':
            return Adam(opt_lr)
        if opt_name == 'RMSprop':
            return RMSprop(opt_lr)
        raise NotImplementedError("Unkonown optimizer: " + opt_name)

    def _fit(self, x_state, x_action, y_value):
        with self._lock_tf:
            with self.tf_graph.as_default():
                batch_size = self.batch_size
                if len(y_value) < 2*batch_size:
                    batch_size = len(y_value) # Still treat it as 1 batch if slightly above

                self.critic_model.fit([x_state, x_action], [y_value], batch_size=batch_size, epochs=1, verbose=0)
                self.actor_critic_model.fit(x_state, y_value, batch_size=batch_size, epochs=1, verbose=0) # y is just placeholder here

    def _predict_action_batch(self, states):
        with self._lock_tf:
            with self.tf_graph.as_default():
                return self.actor_model.predict(states)

    def predict_action(self, state):
        return self._predict_action_batch(state.reshape(1, self.state_size)).flatten()

    def _predict_value_batch(self, states, target=False):
        with self._lock_tf:
            with self.tf_graph.as_default():
                if target:
                    return self.actor_critic_target.predict(states)
                else:
                    return self.actor_critic_model.predict(states)

    def predict_value(self, state):
        return self._predict_value_batch(state.reshape(1, self.state_size), target=False).flatten()[0]

    def _updateTargetModel(self, tau):
        with self._lock_tf:
            with self.tf_graph.as_default():
                new_weights = [t * (1 - tau) + m * (tau)
                               for (t, m)
                               in zip(
                                   self.actor_critic_target.get_weights(),
                                   self.actor_critic_model.get_weights())]
                self.actor_critic_target.set_weights(new_weights)

    def observe(self, data_sequence):
        """
        data_sequence: [[state, action, reward, next_state, _]], ...]
            Consecutive sequence of steps for n-step training
        """
        train_batch = []
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

        # Train
        if len(train_batch) > 0:
            self._train(train_batch)
            self._updateTargetModel(self.target_tau)

    def _train(self, batch_sequences):
        """
        batch_sequences:
            List of sequences of consecutive N (or less) steps
            [
                [[state, action, reward, next_state, _], ..., (step n)]
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
        next_states = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        next_states_v = self._predict_value_batch(next_states, target=True)

        # Build x and y

        x_state = np.zeros((n, self.state_size))
        x_action = np.zeros((n, self.action_size))
        y_value = np.zeros((n, 1))
        cum_reward = 0

        for i in range(n-1, -1, -1):
            [state, action, reward, next_state, _] = batch[i]

            if is_sequence_end[i]:
                # Need to reinitialize target reward from prediction
                if next_state is None:
                    cum_reward = 0
                else:
                    cum_reward = next_states_v[i]

            cum_reward = cum_reward * self.gamma + reward

            x_state[i] = state
            x_action[i] = action
            y_value[i][0] = cum_reward

        self._fit(x_state, x_action, y_value)
