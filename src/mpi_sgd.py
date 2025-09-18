from mpi4py import MPI
import numpy as np

class SGD:
    def __init__(self, params, lr=0.01, comm=MPI.COMM_WORLD):
        self.params = params
        self.lr = lr
        self.comm = comm
        self.params = params
        self.size = comm.Get_size()

    def step(self):
        for p in self.params:
            self.comm.Allreduce(MPI.IN_PLACE, p.grad, op=MPI.SUM)
            p.grad /= self.size  # avg_grad
            p.val -= self.lr * p.grad
        

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
