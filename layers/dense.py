import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        self.grad_weights = np.dot(self.input_data.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        return grad_input
