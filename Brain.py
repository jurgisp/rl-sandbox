
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import threading

class OnlineBrain:
    _lock_tf = threading.Lock()
    _lock_mem = threading.Lock()

    def __init__(self, 
                 env,                  
                 layer1_size=64,
                 layer2_size=0,
                 opt_name='Adam',
                 opt_lr=0.001,
                 opt_loss=keras.losses.mse,
                 minibatch_size=32,
                 target_freq=10000,
                 train_freq=32,                 
                 # Online
                 train_nsteps=1,
                 sarsa=False):

        self.parameters = ['layer1_size', 'layer2_size', 'opt_lr', 
            'minibatch_size', 'target_freq', 'train_freq', 'train_nsteps', 'sarsa']
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.gamma = env.gamma
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_loss = opt_loss
        self.minibatch_size = minibatch_size
        self.target_freq = target_freq
        self.train_freq  = train_freq
        self.train_nsteps = train_nsteps
        self.sarsa = sarsa

        self.model = self._createModel()
        self.target_model = self._createModel()
        self.tf_graph = tf.get_default_graph()
        self.memory = []
        self.target_update_counter = 0

    def get_parameters(self):
        return dict([(p, getattr(self, p)) for p in self.parameters])

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
                self.model.fit(x, y, batch_size=self.minibatch_size, epochs=1, verbose=0)

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

    def observe(self, data_batch):
        train_batch = []
        update_target = False

        with self._lock_mem:
            self.memory.extend(data_batch)

            if len(self.memory) >= self.train_freq:                
                train_batch = self.memory
                self.memory = []

                self.target_update_counter += len(train_batch)
                if self.target_update_counter >= self.target_freq:
                    self.target_update_counter = 0
                    update_target = True

        if len(train_batch) > 0:
            self._train(train_batch)

        if update_target:
            self._updateTargetModel()

    def _train(self, batch):

        n = len(batch)
        x = np.zeros((n, self.state_size))
        y = np.zeros((n, self.action_size))

        no_state = np.zeros(self.state_size)
        next_states = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        next_states_qtarget = self._predictBatch(next_states, target=True)

        cum_reward = 0
        last_state = no_state
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
                    qtarget = next_states_qtarget[i]
                    cum_reward = (
                        qtarget[next_action]
                        if self.sarsa else
                        np.max(qtarget)
                    )

            cum_reward = cum_reward * self.gamma + reward
            target = np.copy(Q)
            target[action] = cum_reward

            x[i] = state
            y[i] = target

            last_state = state
            n_steps += 1

        self._fit(x, y)
