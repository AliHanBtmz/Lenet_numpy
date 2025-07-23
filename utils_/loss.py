import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, pred, target):
        self.pred = np.clip(pred, 1e-12, 1.0)
        self.target = target

        loss = -np.sum(target * np.log(self.pred)) / pred.shape[0]
        return loss

    def backward(self):
        batch_size = self.targets.shape[0]
        grad_input = (self.predictions - self.targets) / batch_size
        return grad_input
