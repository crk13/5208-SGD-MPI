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
    def __init__(self, params, lr=0.1, comm=None):
        self.params = params
        self.lr = lr
        self.comm = MPI.COMM_WORLD if comm is None else comm
    
    # def step(self, local_batch_size):
    #     global_batch_size_sum = self.comm.allreduce(local_batch_size, MPI.SUM)
    #     if global_batch_size_sum == 0:
    #         return
        
    #     for p in self.params:

    #         global_grad_sum = self.comm.allreduce(p.grad, MPI.SUM)

            
            
    #         if global_batch_size_sum > 0:
    #             p.grad = global_grad_sum / global_batch_size_sum
    #             p.val -= self.lr * p.grad

    def step(self, local_batch_size):
        # global sample count
        global_bs = self.comm.allreduce(local_batch_size, op=MPI.SUM)
        if global_bs == 0:
            return

        # 1) 收集本地 grads（为 None 或 zeros 的，都需要转成数组）
        grads_flat_parts = []
        shapes = []
        sizes = []
        for p in self.params:
            g = p.grad
            if g is None:
                g = np.zeros_like(p.val)
            g = np.asarray(g, dtype=np.float64)
            shapes.append(g.shape)
            sizes.append(g.size)
            grads_flat_parts.append(g.ravel())

        if len(grads_flat_parts) == 0:
            return

        local_flat = np.concatenate(grads_flat_parts).astype(np.float64)

        # 2) 全局求和一次
        global_flat = np.zeros_like(local_flat)
        self.comm.Allreduce(local_flat, global_flat, op=MPI.SUM)

        # 3) 取平均并拆回
        global_avg_flat = global_flat / float(global_bs)
        idx = 0
        for p, sh, sz in zip(self.params, shapes, sizes):
            chunk = global_avg_flat[idx: idx + sz].reshape(sh)
            idx += sz
            # 更新 p.grad 与 p.val（更新 in-place）
            p.grad = chunk.copy()
            p.val -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()



class HybridSafeSGD:
    def __init__(self, params, lr=0.1, comm = MPI.COMM_WORLD):
        self.params = params
        self.lr = lr
        self.comm = comm
        # MPI.COMM_TYPE_SHARED是MPI预定义常量，按照共享同一块物理内存（同一台机器）的进程，把它们分到一个通信器里。
        self.local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)

    # def step_vm(self, local_batch_size):
    #     for p in self.params:
    #         thisvm_grad_sum = self.local_comm.allreduce(p.grad, MPI.SUM)
    #         thisvm_batch_size_sum = self.local_comm.allreduce(local_batch_size, MPI.SUM)

    #         if thisvm_batch_size_sum > 0:
    #             p.grad = thisvm_grad_sum / thisvm_batch_size_sum
    #             p.val -= self.lr * p.grad
    
    # def step_glob(self, local_batch_size):
    #     for p in self.params:
    #         psum = self.comm.allreduce(p.val, MPI.SUM)
    #         gsum = self.comm.allreduce(p.grad, MPI.SUM)
    #         bsum = self.comm.allreduce(local_batch_size, MPI.SUM)
    #         if bsum > 0:
    #             p.grad = gsum / bsum
    #             p.val = psum / bsum - self.lr * p.grad

    def step_vm(self, local_batch_size):
        # 本地通信：收集本机进程的梯度
        grads_flat_parts, shapes, sizes = [], [], []
        for p in self.params:
            g = p.grad if p.grad is not None else np.zeros_like(p.val)
            g = np.asarray(g, dtype=np.float64)
            shapes.append(g.shape)
            sizes.append(g.size)
            grads_flat_parts.append(g.ravel())
        local_flat = np.concatenate(grads_flat_parts)
        
        # 本地 allreduce
        local_sum_flat = np.zeros_like(local_flat)
        self.local_comm.Allreduce(local_flat, local_sum_flat, op=MPI.SUM)
        local_avg_flat = local_sum_flat / float(local_batch_size)

        # 拆回
        idx = 0
        for p, sh, sz in zip(self.params, shapes, sizes):
            chunk = local_avg_flat[idx:idx+sz].reshape(sh)
            idx += sz
            p.grad = chunk.copy()
            p.val -= self.lr * p.grad

    def step_glob(self, local_batch_size):
        # 全局通信：收集所有节点梯度
        grads_flat_parts, shapes, sizes = [], [], []
        for p in self.params:
            g = p.grad if p.grad is not None else np.zeros_like(p.val)
            g = np.asarray(g, dtype=np.float64)
            shapes.append(g.shape)
            sizes.append(g.size)
            grads_flat_parts.append(g.ravel())
        local_flat = np.concatenate(grads_flat_parts)

        # 全局 allreduce
        global_flat = np.zeros_like(local_flat)
        self.comm.Allreduce(local_flat, global_flat, op=MPI.SUM)

        # 拆回
        global_bs = self.comm.allreduce(local_batch_size, MPI.SUM)
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


class AdamSafeSGD:
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

    def step_glob(self, local_batch_size):
        grads_flat_parts, shapes, sizes = [], [], []
        for p in self.params:
            g = p.grad if p.grad is not None else np.zeros_like(p.val)
            g = np.asarray(g, dtype=np.float64)
            shapes.append(g.shape)
            sizes.append(g.size)
            grads_flat_parts.append(g.ravel())
        local_flat = np.concatenate(grads_flat_parts)

        # 全局梯度平均
        global_flat = np.zeros_like(local_flat)
        self.comm.Allreduce(local_flat, global_flat, op=MPI.SUM)
        global_bs = self.comm.allreduce(local_batch_size, MPI.SUM)
        global_avg_flat = global_flat / float(global_bs)

        # 拆回
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
