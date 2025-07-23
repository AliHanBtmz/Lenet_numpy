import numpy as np


class PoolLayer:
    def __init__(self, input_shape, pool_size, stride=1, padding=0):
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

    def pad_input(self, input_data):
        if self.padding > 0:
            return np.pad(
                input_data,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )
        return input_data

    def calculate_output_shape(self, h, w):
        out_h = (h - self.pool_size + 2 * self.padding) // self.stride + 1
        out_w = (w - self.pool_size + 2 * self.padding) // self.stride + 1
        return out_h, out_w

    def forward(self, input_data):
        self.input_data = self.pad_input(input_data)
        batch_size, channels, in_h, in_w = self.input_data.shape
        self.out_h, self.out_w = self.calculate_output_shape(in_h, in_w)

        output = np.zeros((batch_size, channels, self.out_h, self.out_w))
        self.mask = np.zeros_like(self.input_data)

        for i in range(self.out_h):
            for j in range(self.out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                region = self.input_data[:, :, h_start:h_end, w_start:w_end]
                out, mask = self.pool_fn(region)
                output[:, :, i, j] = out
                self.mask[:, :, h_start:h_end, w_start:w_end] += mask

        return output

    def backward(self, d_out):
        d_input = np.zeros_like(self.input_data)
        for i in range(self.out_h):
            for j in range(self.out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                grad = d_out[:, :, i, j][:, :, None, None]
                d_input[:, :, h_start:h_end, w_start:w_end] += (
                    self.mask[:, :, h_start:h_end, w_start:w_end] * grad
                )

        if self.padding > 0:
            return d_input[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        return d_input

    def pool_fn(self, region):
        raise NotImplementedError("Must be implement sub classes")


class MaxPool(PoolLayer):
    def pool_fn(self, region):
        max_value = np.max(region, axis=(2, 3), keepdims=True)
        mask = (region == max_value).astype(float)
        return max_value.squeeze(), mask


class AveragePool(PoolLayer):
    def pool_fn(self, region):
        avg_val = np.mean(region, axis=(2, 3), keepdims=True)
        mask = np.ones_like(region) / (self.pool_size * self.pool_size)
        return avg_val.squeeze(), mask
