import numpy as np


class Flatten:
    def __init__(self):
        self.output_shape = None

    def forward(self, input_data):
        self.output_shape = input_data.shape
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.output_shape)
