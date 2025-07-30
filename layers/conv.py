import numpy as np


class Conv2D:
    def __init__(self, input_shape, filter_size, num_filters, stride=1, padding=0):
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding

        self.weights = (
            np.random.randn(num_filters, input_shape[0], filter_size, filter_size)
            * 0.01
        )
        self.biases = np.zeros((num_filters, 1))

    def forward(self, input_data):
        self.input_data = input_data
        if self.padding > 0:
            input_data = np.pad(
                input_data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )

        batch_size, _, in_height, in_width = input_data.shape
        out_height = (
            in_height - self.filter_size + 2 * self.padding
        ) // self.stride + 1
        out_width = (in_width - self.filter_size + 2 * self.padding) // self.stride + 1

        output = np.zeros((batch_size, self.num_filters, out_height, out_width))

        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.filter_size
                        w_end = w_start + self.filter_size

                        region = input_data[b, :, h_start:h_end, w_start:w_end]
                        # Eğer pencere boyutu filter_size x filter_size değilse atla
                        if region.shape != self.weights[f].shape:
                            continue
                        output[b, f, h, w] = (
                            np.sum(region * self.weights[f]) + self.biases[f, 0]
                        )

        return output

    def backward(self, output_gradient):
        input_data = self.input_data
        if self.padding > 0:
            input_data = np.pad(
                input_data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )

        batch_size, _, _, _ = input_data.shape
        _, _, out_height, out_widht = output_gradient.shape

        d_input = np.zeros_like(input_data)
        d_weights = np.zeros_like(self.weights)
        d_biases = np.zeros_like(self.biases)

        for b in range(batch_size):
            for f in range(self.num_filters):
                for h in range(out_height):
                    for w in range(out_widht):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.filter_size
                        w_end = w_start + self.filter_size

                        region = input_data[b, :, h_start:h_end, w_start:w_end]
                        if region.shape != self.weights[f].shape:
                            continue
                        d_weights[f] += output_gradient[b, f, h, w] * region
                        d_biases[f] += output_gradient[b, f, h, w]
                        d_input[b, :, h_start:h_end, w_start:w_end] += (
                            output_gradient[b, f, h, w] * self.weights[f]
                        )
        if self.padding > 0:
            d_input = d_input[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]

        self.d_weights = d_weights
        self.d_biases = d_biases
        return d_input

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases
