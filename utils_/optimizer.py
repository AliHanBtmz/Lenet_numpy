import numpy as np


class SGD:
    def __init__(self, parameters, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.parameters = parameters

    def step(self):
        for param, grad in self.parameters:
            if grad is not None:
                param -= self.learning_rate * grad


class Adam:
    def __init__(
        self,
        parameters,
        epsilon=1e-8,
        beta_1=0.9,
        beta_2=0.999,
        learning_rate=0.01,
    ):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.v = {}
        self.m = {}
        self.t = 0

        for i, (param, _) in enumerate(parameters):
            self.v[i] = np.zeros_like(param)
            self.m[i] = np.zeros_like(param)

    def step(self):
        self.t += 1

        for i, (param, grad) in enumerate(self.parameters):
            if grad is None:
                continue
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (grad**2)

            m_hat = self.m[i] / (1 - self.beta_1**self.t)
            v_hat = self.v[i] / (1 - self.beta_2**self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
