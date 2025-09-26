# mpirun --use-hwthread-cpus -np 4 python test/testcopy.py
# /usr/bin/mpirun -np 12 --hostfile hostfile /home/76836/miniconda3/envs/mpi/bin/python test/testcopy.py


import numpy as np
from mpi4py import MPI
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.model import Linear, Sequential, ReLU, Tanh, Sigmoid, MSELoss, RMSELoss
from source.mpi_sgd import SafeSGD, HybridSafeSGD
from source.data import mpi_read_data
from source.mpi_train import global_train, global_rmse_scaled

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ====================== 数据加载 ======================
n_fea = 19
training, test, ntrain, ntest = mpi_read_data("data/processed/dataall_normalized.parquet", n_fea)
print(f"Loading Done! [Rank {rank}] Finished loading data. Local training samples shape: {training.shape}, test samples shape: {test.shape}")

X_local = training[:, :-1]
y_local = training[:, -1].reshape(-1, 1)
X_test = test[:, :-1]
y_test = test[:, -1].reshape(-1, 1)

y_min, y_max = y_local.min(), y_local.max()
y_local_scaled = (y_local - y_min) / (y_max - y_min)




# ====================== 模型初始化 ======================
hidden = 20
lr = 0.005
np.random.seed(42 + rank)


model = Sequential(
    Linear(n_fea, hidden), 
    Tanh(),
    Linear(hidden, 1)                  
)
optimizer = HybridSafeSGD(model.params, lr, comm)
loss_fn = MSELoss()
# if rank == 0:
#     print(f"[Rank {rank}] Model and optimizer initialized")


# ====================== 训练 ======================
batch_size = 256
epochs = 20
shuffle = True
glob_interval=1000
for epoch in range(epochs):

    epoch_loss = global_train(model, optimizer, X_local, y_local_scaled, loss_fn, batch_size, shuffle, glob_interval)
    train_rmse = global_rmse_scaled(model, X_local, y_local, y_max, y_min)
    test_rmse = global_rmse_scaled(model, X_test, y_test, y_max, y_min)

    if rank == 0:
        print(f"Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

print(f"done! [Rank {rank}]")