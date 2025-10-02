from mpi4py import MPI
import numpy as np
from source.para import MPITensor

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

class MPISGD:
    def __init__(self, params, lr=0.1, comm=None):
        self.params = params
        self.lr = lr
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def step(self, local_batch_size):
        # global sample count
        global_bs = self.comm.allreduce(local_batch_size, MPI.SUM)
        if global_bs == 0:
            return

        grads_flat_parts = []
        shapes = []
        sizes = []
        for p in self.params:
            g = p.grad if p.grad is not None else np.zeros_like(p.val)
            g = np.asarray(g, dtype=np.float64)
            shapes.append(g.shape)
            sizes.append(g.size)
            grads_flat_parts.append(g.ravel())
        if len(grads_flat_parts) == 0:
            return
    
        local_flat = np.concatenate(grads_flat_parts).astype(np.float64)
        global_flat = self.comm.allreduce(local_flat, MPI.SUM)
        global_avg_flat = global_flat / float(global_bs)
        idx = 0
        for p, sh, sz in zip(self.params, shapes, sizes):
            chunk = global_avg_flat[idx: idx + sz].reshape(sh)
            idx += sz
            p.grad = chunk.copy()
            p.val -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()



class HybridMPISGD:
    def __init__(self, params, lr=0.1, comm = MPI.COMM_WORLD):
        self.params = params
        self.lr = lr
        self.comm = comm
        # MPI.COMM_TYPE_SHARED是MPI预定义常量，按照共享同一块物理内存（同一台机器）的进程，把它们分到一个通信器里。
        self.local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    def step_vm(self, local_batch_size):
        local_bs = self.local_comm.allreduce(local_batch_size, MPI.SUM)
        if local_bs == 0:
            return
        
        grads_flat_parts, shapes, sizes = [], [], []
        for p in self.params:
            g = p.grad if p.grad is not None else np.zeros_like(p.val)
            g = np.asarray(g, dtype=np.float64)
            shapes.append(g.shape)
            sizes.append(g.size)
            grads_flat_parts.append(g.ravel())
        if len(grads_flat_parts) == 0:
            return
        
        local_flat = np.concatenate(grads_flat_parts).astype(np.float64)
        local_sum_flat = self.local_comm.allreduce(local_flat, MPI.SUM)
        local_avg_flat = local_sum_flat / float(local_bs)

        idx = 0
        for p, sh, sz in zip(self.params, shapes, sizes):
            chunk = local_avg_flat[idx:idx+sz].reshape(sh)
            idx += sz
            p.grad = chunk.copy()
            p.val -= self.lr * p.grad

    def step_glob(self, local_batch_size):
        global_bs = self.comm.allreduce(local_batch_size, MPI.SUM)
        if global_bs == 0:
            return
        
        grads_flat_parts, shapes, sizes = [], [], []
        for p in self.params:
            g = p.grad if p.grad is not None else np.zeros_like(p.val)
            g = np.asarray(g, dtype=np.float64)
            shapes.append(g.shape)
            sizes.append(g.size)
            grads_flat_parts.append(g.ravel())
        if len(grads_flat_parts) == 0:
            return
        
        local_flat = np.concatenate(grads_flat_parts).astype(np.float64)
        global_flat = self.comm.allreduce(local_flat, MPI.SUM)
        global_avg_flat = global_flat / float(global_bs)

        idx = 0
        for p, sh, sz in zip(self.params, shapes, sizes):
            chunk = global_avg_flat[idx:idx+sz].reshape(sh)
            idx += sz
            p.grad = chunk.copy()
            p.val -= self.lr * p.grad
             
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class AdamMPISGD:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, comm=MPI.COMM_WORLD):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.comm = comm

        # 初始化动量
        self.m = [np.zeros_like(p.val) for p in self.params]
        self.v = [np.zeros_like(p.val) for p in self.params]
        self.t = 0  # timestep

    def step(self, local_batch_size):
        global_bs = self.comm.allreduce(local_batch_size, MPI.SUM)
        if global_bs == 0:
            return

        grads_flat_parts, shapes, sizes = [], [], []
        for p in self.params:
            g = p.grad if p.grad is not None else np.zeros_like(p.val)
            g = np.asarray(g, dtype=np.float64)
            shapes.append(g.shape)
            sizes.append(g.size)
            grads_flat_parts.append(g.ravel())
        if len(grads_flat_parts) == 0:
            return
        
        local_flat = np.concatenate(grads_flat_parts)
        global_flat = self.comm.allreduce(local_flat, MPI.SUM)
        global_avg_flat = global_flat / float(global_bs)

        # reshape
        idx = 0
        self.t += 1
        for i, (p, sh, sz) in enumerate(zip(self.params, shapes, sizes)):
            chunk = global_avg_flat[idx:idx+sz].reshape(sh)
            idx += sz

            # Adam 更新
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * chunk
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (chunk ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.val -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.grad = chunk.copy()  # 更新 grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


if __name__ == "__main__":
    para = MPITensor(np.array([1.0, 2.0], dtype = np.float64))
    optim = MPISGD([para], 0.1, comm)

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
