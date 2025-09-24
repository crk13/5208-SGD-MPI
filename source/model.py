import numpy as np
from source.para import MPITensor

# class Module:
#     def __init__(self):
#         self.x = None

#     def forward(self, x):
#         raise NotImplementedError
    
#     def backward(self, dLdy):
#         raise NotImplementedError

class Linear:
    def __init__(self, indim, outdim):
        # super().__init__()
        self.W = MPITensor(np.random.randn(indim, outdim))
        self.b = MPITensor(np.zeros(outdim))
        self.x = None
        self.params = [self.W, self.b]

    def forward(self, x):
        self.x = x
        return x@self.W.val + self.b.val
    
    def backward(self, dLdy):
        self.W.grad += self.x.T @ dLdy
        self.b.grad += np.sum(dLdy, axis=0)
        #  # 直接覆盖梯度，而不是累积？？？
        # self.W.grad = self.x.T @ dLdy
        # self.b.grad = dLdy.sum(axis=0, keepdims=True)
        return dLdy @ self.W.val.T

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        # embedding 矩阵初始化
        self.weight = MPITensor(np.random.randn(num_embeddings, embedding_dim) * 0.01)
        self.input = None  # 保存索引
        self.params = [self.weight]

    def forward(self, x):
        """
        x: shape (batch_size,) 或 (batch_size, seq_len)
        返回 shape: (batch_size, embedding_dim) 或 (batch_size, seq_len, embedding_dim)
        """
        self.input = x
        return self.weight.val[x]  # 利用 numpy 整数索引直接查表

    def backward(self, dLdy):
        """
        dLdy: shape 与 forward 输出一致
        """
        # 初始化梯度矩阵为0
        grad = np.zeros_like(self.weight.val)
        
        # 支持 batch 的索引累加梯度
        if dLdy.ndim == 2:  # (batch_size, embedding_dim)
            for i, idx in enumerate(self.input):
                grad[idx] += dLdy[i]
        elif dLdy.ndim == 3:  # (batch_size, seq_len, embedding_dim)
            for i in range(dLdy.shape[0]):
                for j in range(dLdy.shape[1]):
                    grad[self.input[i, j]] += dLdy[i, j]
        else:
            raise ValueError("Unsupported dLdy shape")
        
        self.weight.grad += grad
        return None  # embedding 层没有输出对输入的梯度，因为输入是索引

class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    @property
    def params(self):
        all_params = []
        for layer in self.layers:
            if hasattr(layer, 'params'):
                all_params.extend(layer.params)
        return all_params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def backward(self, dLdy):
        for layer in reversed(self.layers):
            dLdy = layer.backward(dLdy)
        return dLdy
    
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, dLdy):
        return dLdy * (self.x > 0)
    
class Tanh:
    def forward(self, x):
        self.x = x
        return np.tanh(self.x)
    
    def backward(self, dLdy):
        return dLdy * (1 - np.tanh(self.x) ** 2)
    
class Sigmoid:
    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))
    
    def backward(self, dLdy):
        sig = 1 / (1 + np.exp(-self.x))
        return dLdy * sig * (1 - sig)
    
class MSELoss:
    def __init__(self):
        self.yhat = None
        self.y = None

    def forward(self, yhat, y):
        self.yhat = yhat
        self.y = y
        loss = 0.5 * np.mean((yhat - y) ** 2)
        return loss
    
    def backward(self):
        B = self.y.shape[0]
        return (self.yhat - self.y) / B