
import numpy as np

class LinearBrain:
    def __init__(self, env, coeffs):
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.coeffs = coeffs

    def train(self, x, y, epoch=1, verbose=0):
        return

    def predictBatch(self, state, target=False):
        raise NotImplementedError("LinearBrain is not used to training!")

    def predictOne(self, state, target=False):
        y = np.matmul(self.coeffs, state)
        return [-y, y]