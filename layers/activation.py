import numpy as np


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, grad_output):
        relu_derivate = self.input > 0
        return grad_output * relu_derivate


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, input_data):
        exp_data = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.output = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output):

        batch_size, num_classes = self.output.shape
        grad_input = np.zeros_like(grad_output)

        for i in range(batch_size):
            y = self.output[i].reshape(-1, 1)
            jacobian = np.diagflat(y) - y @ y.T
            grad_input[i] = jacobian @ grad_output[i]
        return grad_input


class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, input_data):
        self.output = np.tanh(input_data)
        return self.output

    def backward(self, input_data, grad_output):
        tanh_derivative = 1.0 - self.output**2
        return grad_output * tanh_derivative
