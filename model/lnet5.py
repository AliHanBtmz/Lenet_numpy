from layers.conv import Conv2D
from layers.pool import MaxPool, AveragePool
from layers.activation import ReLU, Softmax, Tanh
from layers.flatten import Flatten
from layers.dense import Dense


class LNet5:
    def __init__(self):
        self.layers = [
            Conv2D(
                input_shape=(1, 28, 28),
                filter_size=5,
                num_filters=6,
                stride=1,
                padding=2,
            ),
            Tanh(),
            MaxPool(input_shape=(6, 28, 28), pool_size=2, stride=2),
            Conv2D(
                input_shape=(6, 14, 14),
                filter_size=5,
                num_filters=16,
                stride=1,
                padding=0,
            ),
            Tanh(),
            MaxPool(input_shape=(16, 10, 10), pool_size=2, stride=2),
            Flatten(),
            Dense(input_size=16 * 5 * 5, output_size=120),
            Tanh(),
            Dense(input_size=120, output_size=84),
            Tanh(),
            Dense(input_size=84, output_size=10),
            Softmax(),
        ]

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                grad_output = layer.backward(grad_output)
        return grad_output

    def get_parameters_and_grad(self):
        pass
