# src/grad_descent.py
import numpy as np
# 依赖于model.py
from src.model import Linear, ReLU, Tanh, Sigmoid, Sequential, MSELoss

# ---- 约定：返回/接收的都是 numpy 数组，不暴露 MPITensor 给 mpi.py ----

def _activation(name: str):
    if name == "relu": return ReLU()
    if name == "tanh": return Tanh()
    if name == "sigmoid": return Sigmoid()
    raise ValueError(f"Unknown activation: {name}")

def init_params(in_dim: int, hidden: int, activation: str, seed: int):
    rng = np.random.default_rng(seed)
    # 用 float32 初始化更稳
    L1 = Linear(in_dim, hidden)     # 里面会创建 MPITensor
    L2 = Linear(hidden, 1)
    # 重置权重为 float32 的 Xavier/He（可选优化）
    L1.W.val = (rng.standard_normal((in_dim, hidden)).astype(np.float32) / np.sqrt(in_dim))
    L1.b.val = np.zeros((hidden,), dtype=np.float32)
    L2.W.val = (rng.standard_normal((hidden, 1)).astype(np.float32) / np.sqrt(hidden))
    L2.b.val = np.zeros((1,), dtype=np.float32)

    model = Sequential(L1, _activation(activation), L2)
    loss_fn = MSELoss()
    # 返回一个“容器对象”，mpi.py 只把它当 opaque 的 params 用
    return {"model": model, "loss": loss_fn}

def _zero_grads(model):
    # 遍历所有含 params 的层，调用底层 MPITensor.zero_grad()
    for layer in model.layers:
        if hasattr(layer, "params"):
            for p in layer.params:
                p.zero_grad()

def compute_local_grads(params, Xb, Yb):
    """
    返回：
      grads: dict[str, np.ndarray]  —— 将所有参数梯度打平成 numpy 数组字典
      loss: float                   —— 本地 loss 总和（注意：我们这里返回 sum 而不是 mean，便于全局加权）
      count: int                    —— 本地样本数
    """
    model = params["model"]
    loss_fn = params["loss"]

    # 空批：直接零梯度
    if Xb.shape[0] == 0:
        grads = {}
        # 为了和 mpi.py 的聚合一致，返回0损失和0样本
        return grads, 0.0, 0

    # 前向
    yhat = model.forward(Xb.astype(np.float32, copy=False))
    loss = loss_fn.forward(yhat, Yb.astype(np.float32, copy=False))  # 这是 mean loss * 0.5

    # 反向
    _zero_grads(model)
    dLdy = loss_fn.backward()                 # 形状 (B,1)
    model.backward(dLdy)

    # 收集梯度为 numpy 字典（与 mpi.py 的 allreduce 兼容）
    grads = {}
    idx = 0
    for layer in model.layers:
        if hasattr(layer, "params"):
            for p in layer.params:
                # 键名可按层序与参数名拼接，保证唯一
                key = f"p{idx}"
                grads[key] = p.grad.astype(np.float32, copy=False)
                idx += 1

    # 注意：mpi.py 里做的是“全局求和后 / 全局样本数”，
    # 所以这里返回的是 “本地 sum loss 和 本地样本数”，而不是 mean。
    loss_sum = float(loss * Xb.shape[0] * 2.0)  # 因为 loss_fn 用了 0.5*mean，这里乘回去得到 MSE 的 sum
    return grads, loss_sum, int(Xb.shape[0])

def apply_grads(params, grads_avg: dict, lr: float) -> None:
    model = params["model"]
    idx = 0
    for layer in model.layers:
        if hasattr(layer, "params"):
            for p in layer.params:
                g = grads_avg.get(f"p{idx}")
                if g is not None:
                    p.apply_(lr, g)
                idx += 1

def predict(params, Xb):
    model = params["model"]
    return model.forward(Xb.astype(np.float32, copy=False)).reshape(-1, 1)