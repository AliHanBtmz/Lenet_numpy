import numpy as np


def load_mnist_images(path):
    with open(path, "r+b") as f:
        data = f.read()

        num_images = int.from_bytes(data[4:8], byteorder="big")
        rows = int.from_bytes(data[8:12], byteorder="big")
        cols = int.from_bytes(data[12:16], byteorder="big")

        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        images = images.reshape(num_images, 1, rows, cols)  # 1 = kanal
        # images = images.astype(np.float32) / 255.0  # normalize et
        return images


def load_mnist_label(path):
    with open(path, "rb") as f:
        data = f.read()
        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        return labels


def one_hot_encode(labels, num_classes=10):

    return np.eye(num_classes)[labels]
