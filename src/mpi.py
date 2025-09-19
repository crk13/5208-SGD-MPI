# src/mpi.py
import argparse
import numpy as np
from mpi4py import MPI
#mpi.py (训练循环) → 调用 grad_descent.py → 用到 model.py (网络结构) → 用到 para.py (参数张量)
# ---- 与另一位同学约定的接口（在 grad_descent.py 中实现） ----
# - init_params(in_dim:int, hidden:int, activation:str, seed:int) -> params(any)
# - compute_local_grads(params, Xb, yb) -> (grads:dict[str,np.ndarray], loss:float, count:int)
# - apply_grads(params, grads_avg:dict[str,np.ndarray], lr:float) -> None
# - predict(params, Xb) -> yhat (for RMSE eval)
import src.grad_descent as GD  # 另一位同学提供

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def log0(msg: str):
    # log0 只在 rank 0 打印，避免多进程重复输出
    if rank == 0:
        print(msg, flush=True)

def load_npz_local(path: str):
    # 约定：预处理脚本把数据存成 npz：X, y（y 为列向量 shape=(N,1)）
    data = np.load(path, allow_pickle=True)
    # 兼容旧键名 df_X/df_y
    X = data["X"] if "X" in data else data["df_X"]
    y = data["y"] if "y" in data else data["df_y"]
    return X, y


def load_npz_distributed(path: str):
    """rank0 读取完整 npz，然后按 rank 均匀切片后 scatter 给各个进程。
    返回：X_local, y_local, N_total
    """
    if rank == 0:
        data = np.load(path, allow_pickle=True)
        X = data["X"] if "X" in data else data["df_X"]
        y = data["y"] if "y" in data else data["df_y"]
        # 统一 dtype/形状
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        N_total = int(X.shape[0])
        # 按 rank 均匀切片（与 shard_indices 一致的近乎均匀分片）
        X_chunks = np.array_split(X, size)
        y_chunks = np.array_split(y, size)
    else:
        X_chunks = None
        y_chunks = None
        N_total = None
    # 分发各自分片
    X_local = comm.scatter(X_chunks, root=0)
    y_local = comm.scatter(y_chunks, root=0)
    # 广播全局样本总数（用于 train_epoch 中的全局采样与映射）
    N_total = comm.bcast(N_total, root=0)
    return X_local, y_local, N_total

def shard_indices(n_samples: int, world_size: int, r: int):
    # world_size：进程总数（即 MPI 的 rank 总数）；r：当前进程的 rank 编号。
    # 给定样本总数 n_samples，划分给 world_size 个 rank，返回第 r 个 rank 的样本索引范围 [start, end)
    # 近乎均匀分片，每个 rank 负责 [start:end)
    base = n_samples // world_size # 计算每个进程至少分到的样本数 
    rem = n_samples % world_size # 计算剩余未均分的样本数
    start = r * base + min(r, rem) # 每个进程的起始索引
    end = start + base + (1 if r < rem else 0) # 每个进程的结束索引，前 rem 个进程会多分到 1 个样本
    return start, end

# ----------- 广播批索引 + 全局求和 -----------
def bcast_batch_indices(global_bs_idx):
    # 用于广播数据（这里是 batch 索引）到所有进程。
    # global_bs_idx: rank0 采样的全局 batch 索引（list/ndarray），其他 rank 传 None
    # rank0 负责采样并广播，只有 rank 0 负责采样，其他进程接收
    return comm.bcast(global_bs_idx, root=0) # 从 rank 0 广播变量 global_bs_idx，所有进程收到一样的内容。

def allreduce_sum_array(x: np.ndarray):
    # 用于对所有进程的数组做“全局求和”操作
    # 每个进程传入自己的数组 x，所有进程得到所有数组元素逐项相加的结果
    y = np.zeros_like(x) # 创建一个同样 shape 的零数组
    comm.Allreduce(x, y, op=MPI.SUM) # 执行 Allreduce 操作，将 x 的值累加到 y 中
    return y

def allreduce_sum_scalar(x: float):
    # 用于对所有进程的标量做“全局求和”操作
    # 每个进程传入自己的标量 x，所有进程得到所有标量相加的结果
    return comm.allreduce(x, op=MPI.SUM)

def train_epoch(params, X_local, y_local, batch_size, lr, sampler_rng, total_N):
    """单个 epoch：
    1) rank0 采样全局 batch 索引（不放数据，只放索引）
    2) 各 rank 就地取子样本
    3) 计算本地梯度与loss
    4) Allreduce 求和 -> 取平均 -> 更新参数
    """
    # 为了简单，这里每个 epoch 用若干个 mini-batch，步数 = ceil(N / batch_size)
    steps = (total_N + batch_size - 1) // batch_size  # 计算本 epoch 需要多少个 mini-batch

    # 先计算一次本地分片范围（移出循环，避免重复计算）
    start_local, end_local = shard_indices(total_N, size, rank)

    # 本 epoch 先做一次全局打乱（全排列），再按 batch 切块；只广播一次，避免每 step 重复广播
    if rank == 0:
        perm = sampler_rng.permutation(total_N)
    else:
        perm = None
    perm = comm.bcast(perm, root=0)

    loss_running = 0.0
    for s in range(steps):
        ## 1. 采样 batch 索引（只在 rank 0）
                # 这里不再每步 random choice；而是从本 epoch 的全局打乱 perm 中切片得到 batch_idx
        batch_idx = perm[s * batch_size : min((s + 1) * batch_size, total_N)]
        # 注意：由于我们已经广播了 perm，这里无需再次广播；若希望显式同步，也可以：
        # batch_idx = bcast_batch_indices(batch_idx)

        ## 2. 广播batch索引： 把 rank 0 采样的索引广播给所有进程，保证大家用的是同一批数据。
        # batch_idx = bcast_batch_indices(batch_idx)  # list/ndarray of indices (global)

        ## 3. 本地分片映射： 每个进程根据自己的分片范围（shard_indices），筛选出本地命中的样本（即全局索引落在本地分片范围内的那些），并转换为本地相对索引。
        # 从本地分片里挑出命中的样本：把全局索引映射到本地片
        # 我们的本地片是 [start:end)
        # 如果某个全局 idx 不在本地范围，就忽略
        # 先算一下自己的范围
        # start_local, end_local = shard_indices(total_N, size, rank)
        # 找命中当前分片的全局索引
        mask = (batch_idx >= start_local) & (batch_idx < end_local)
        local_idx_global = batch_idx[mask]
        # 映射到本地相对索引
        local_idx = local_idx_global - start_local

        ## 4. 本地数据准备： 如果本地有命中样本，则取出对应的 Xb, yb，否则给一个“空批”（shape 正确但无数据）。
        if local_idx.size > 0:
            Xb = X_local[local_idx]
            yb = y_local[local_idx]
        else:
            # 没有命中时，给一个“空批”。compute_local_grads需要能处理空输入（返回0梯度，count=0）
            Xb = X_local[:0]
            yb = y_local[:0]

        ## 5. 本地梯度与损失计算： 让另一位同学的代码计算本地梯度、损失和样本数（count）（只用 numpy，禁止在对方代码里 import MPI）
        grads_loc, loss_loc, cnt_loc = GD.compute_local_grads(params, Xb, yb)

        ## 6. 全局聚合 ---- Allreduce 聚合（非阻塞 Iallreduce 版本）----
        # 1) 聚合标量（loss、count）：先发起非阻塞 allreduce，再等待
        #    注意：mpi4py 的 Iallreduce 需要显式的 send/recv buffer（numpy 数组）
        loss_send = np.array(float(loss_loc), dtype=np.float64)
        loss_recv = np.empty(1, dtype=np.float64)
        req_loss = comm.Iallreduce(loss_send, loss_recv, op=MPI.SUM)

        cnt_send = np.array(int(cnt_loc), dtype=np.int64)
        cnt_recv = np.empty(1, dtype=np.int64)
        req_cnt = comm.Iallreduce(cnt_send, cnt_recv, op=MPI.SUM)

        # 2) 聚合每一个梯度张量：为每个梯度张量发起 Iallreduce，然后统一等待（可与轻量计算重叠）
        grad_recv_bufs = {}
        grad_reqs = []
        for k, g in grads_loc.items():
            # 确保 dtype/shape 一致；采用就地类型（通常 float32）
            g_send = np.asarray(g, dtype=np.float32)
            g_recv = np.empty_like(g_send)
            req = comm.Iallreduce(g_send, g_recv, op=MPI.SUM)
            grad_recv_bufs[k] = g_recv
            grad_reqs.append(req)

        # —— 到这里通信已在后台进行。若有可并行的 CPU 工作，可在此处执行（例如下一个 batch 的索引准备）。——

        # 3) 等待标量与梯度的通信完成
        req_loss.Wait()
        req_cnt.Wait()
        MPI.Request.Waitall(grad_reqs)

        loss_sum = float(loss_recv[0])
        cnt_sum = int(cnt_recv[0])

        # 4) 归一化得到全局平均梯度
        grads_avg = {}
        if cnt_sum > 0:
            inv_cnt = 1.0 / float(cnt_sum)
            for k, g_sum in grad_recv_bufs.items():
                grads_avg[k] = g_sum * inv_cnt
        else:
            # 极端情况下（本批全空）维持零梯度
            for k, g_sum in grad_recv_bufs.items():
                grads_avg[k] = g_sum

        # 5) 每个 rank 同步地应用同一个 grads_avg 更新（使得参数保持一致）
        GD.apply_grads(params, grads_avg, lr)

        # 统计平均损失（除以全局样本数）
        loss_running += (loss_sum / max(cnt_sum, 1))

    return loss_running / steps # 返回本 epoch 的平均损失（所有 mini-batch 的损失均值）

def rmse_eval(params, X_local, y_local, total_N):
    """并行 RMSE：各 rank 计算局部 SSE 与 count，再 allreduce 求和。"""
    if X_local.shape[0] == 0:
        sse_loc = 0.0
        cnt_loc = 0
    else:
        yhat = GD.predict(params, X_local)
        err = (yhat.reshape(-1, 1) - y_local)
        sse_loc = float(np.sum(err * err))
        cnt_loc = int(X_local.shape[0])

    sse = allreduce_sum_scalar(sse_loc)
    cnt = allreduce_sum_scalar(cnt_loc)
    return np.sqrt(sse / max(cnt, 1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/processed/train.npz")
    parser.add_argument("--test", type=str, default="data/processed/test.npz")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu","tanh","sigmoid"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 固定随机种子（rank0 统一种子，广播给大家，确保采样一致）
    if rank == 0:
        seed = args.seed
    else:
        seed = None
    seed = comm.bcast(seed, root=0)
    np.random.seed(seed)
    sampler_rng = np.random.default_rng(seed)

    # 载入训练/测试数据：rank0 读取并分发，各 rank 本地只持有自己的分片
    Xtr_loc, ytr_loc, Ntr = load_npz_distributed(args.train)
    Xte_loc, yte_loc, Nte = load_npz_distributed(args.test)

    # 初始化模型参数（所有 rank 必须一致；各本地分片的列数相同）
    in_dim = Xtr_loc.shape[1]
    params = GD.init_params(in_dim=in_dim, hidden=args.hidden,
                            activation=args.activation, seed=seed)

    # 训练循环
    best_val = None
    patience = 5
    no_improve = 0
    for ep in range(1, args.epochs + 1):
        loss_ep = train_epoch(params, Xtr_loc, ytr_loc, args.batch_size, args.lr, sampler_rng, total_N=Ntr)
        rmse_tr = rmse_eval(params, Xtr_loc, ytr_loc, total_N=Ntr)
        rmse_te = rmse_eval(params, Xte_loc, yte_loc, total_N=Nte)

        if rank == 0:
            log0(f"[Epoch {ep}] loss={loss_ep:.6f}  RMSE(train)={rmse_tr:.4f}  RMSE(test)={rmse_te:.4f}")

        # 早停：以测试集 RMSE 为监控
        metric = rmse_te
        improved = (best_val is None) or (metric < best_val - 1e-6)
        if improved:
            best_val = metric
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log0(f"Early stopping at epoch {ep} (best RMSE={best_val:.4f})")
                break

if __name__ == "__main__":
    main()