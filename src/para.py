import numpy as np

class MPITensor:
    def __init__(self, val):
        self.val = val
        self.grad = np.zeros_like(val)

    def zero_grad(self):
        self.grad = np.zeros_like(self.val)
        