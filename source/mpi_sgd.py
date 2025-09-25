from mpi4py import MPI
import numpy as np
from source.para import MPITensor

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

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


class SafeSGD:
    def __init__(self, params, lr=0.1, comm=MPI.COMM_WORLD):
        self.params = params
        self.lr = lr
        self.comm = comm
        self.size = comm.Get_size()

    def step(self, local_batch_size):
        for p in self.params:
            # local_grad = p.grad if p.grad is not None else np.zeros_like(p.val)
            global_grad = comm.allreduce(p.grad, MPI.SUM)

            global_batch_size = comm.allreduce(local_batch_size, MPI.SUM)
            
            if global_batch_size > 0:
                p.grad = global_grad / global_batch_size
                p.val -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


if __name__ == "__main__":
    para = MPITensor(np.array([1.0, 2.0], dtype = np.float64))
    optim = SafeSGD([para], 0.1, comm)

    if rank == 0:
        para.grad = np.array([0.1, 0.2], dtype = np.float64)
        localb = 10
    else:
        para.grad = np.zeros(2, dtype = np.float64)
        localb = 0

        print(f"[Rank {rank}] Before step: grad={para.grad}")
    optim.step(localb)
    print(f"[Rank {rank}] After step: grad={para.grad}")


# cd /mntt/data/5208-SGD-MPI
# mpirun -np 2 python -m source.mpi_sgd
