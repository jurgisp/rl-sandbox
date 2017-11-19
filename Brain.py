
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop

class Brain:
    def __init__(self, 
                 env, 
                 layer1_size=64,
                 layer2_size=0,
                 opt_name='Adam',
                 opt_lr=0.001,
                 opt_loss=keras.losses.mse):

        self.parameters = ['layer1_size', 'layer2_size', 'opt_lr']
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.opt_name = opt_name
        self.opt_lr = opt_lr
        self.opt_loss = opt_loss

        self.model = self._createModel()
        self.target_model = self._createModel()

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

    def train(self, x, y):
        self.model.fit(x, y, batch_size=len(y), epochs=1, verbose=0)

    def predictBatch(self, s, target=False):
        if target:
            return self.target_model.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predictBatch(s.reshape(1, self.state_size), target=target).flatten()

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_parameters(self):
        return dict([(p, getattr(self, p)) for p in self.parameters])
    