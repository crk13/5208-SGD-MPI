import numpy as np

class MPITensor:
    def __init__(self, val):
        # self.val = val
        self.val = np.asarray(val, dtype=np.float32)   # 确保参数是 float32
        self.grad = np.zeros_like(val)

    def zero_grad(self):
        # self.grad = np.zeros_like(self.val)
        self.grad.fill(0)  # 原地清零，替代 np.zeros_like 重新分配

    def apply_(self, lr, g):
        # 所有参数更新操作都绑定在参数对象（MPITensor）上
        # 假设 g 形状/类型与 self.val 一致
        self.val -= lr * g