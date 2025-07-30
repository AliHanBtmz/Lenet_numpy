from layers.conv import Conv2D
from layers.pool import MaxPool, AveragePool
from layers.activation import ReLU, Softmax, Tanh
from layers.flatten import Flatten
from layers.dense import Dense
import numpy as np


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
            Dense(input_size=16 * 6 * 6, output_size=120),
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
        params_and_grads = []
        for layer in self.layers:
            if isinstance(layer, Conv2D):
                d_weights = layer.d_weights if hasattr(layer, "d_weights") else None
                d_biases = layer.d_biases if hasattr(layer, "d_biases") else None
                params_and_grads.append((layer.weights, d_weights))
                params_and_grads.append((layer.biases, d_biases))
            if isinstance(layer, Dense):
                grad_weights = (
                    layer.grad_weights if hasattr(layer, "grad_weights") else None
                )
                grad_biases = (
                    layer.grad_biases if hasattr(layer, "grad_biases") else None
                )
                params_and_grads.append((layer.weights, grad_weights))
                params_and_grads.append((layer.biases, grad_biases))
        return params_and_grads

    def predict(self, input_data):
        output = self.forward(input_data)
        return np.argmax(output)

    def load_weights(self, path="lenet5_weights.npz"):
        import numpy as np

        data = np.load(path)
        layer_idx = 0
        for layer in self.layers:
            w_key = f"layer{layer_idx}_weights"
            b_key = f"layer{layer_idx}_biases"
            if hasattr(layer, "weights") and w_key in data:
                layer.weights = data[w_key]
            if hasattr(layer, "biases") and b_key in data:
                layer.biases = data[b_key]
            layer_idx += 1
        print(f"Weights loaded from {path}")
