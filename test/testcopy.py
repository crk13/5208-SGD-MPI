# mpirun --use-hwthread-cpus -np 4 python test/testcopy.py

import numpy as np
from mpi4py import MPI
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.model import Linear, Sequential, ReLU, Tanh, Sigmoid, MSELoss, RMSELoss
from source.mpi_sgd import SafeSGD
from source.data import mpi_read_data
from source.mpi_train import global_train, global_rmse_eval

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ====================== 数据加载 ======================
n_fea = 19
training, test, ntrain, ntest = mpi_read_data("data/processed/data4000000.parquet", n_fea)
print(f"Loading Done! [Rank {rank}] Finished loading data. Local training samples: {training.shape[0]}, test samples: {test.shape[0]}")

X_local = training[:, :-1]
y_local = training[:, -1].reshape(-1, 1)
X_test = test[:, :-1]
y_test = test[:, -1].reshape(-1, 1)
print(f"[Rank {rank}] x_local shape: {X_local.shape}, y_local shape: {y_local.shape}")


# ====================== 模型初始化 ======================
hidden = 10
lr = 0.1
np.random.seed(42 + rank)
batch_count = 5

model = Sequential(
    Linear(n_fea, hidden), 
    Tanh(),
    Linear(hidden, 1)                  
)
optimizer = SafeSGD(model.params, lr, comm)
loss_fn = MSELoss()
if rank == 0:
    print(f"[Rank {rank}] Model and optimizer initialized")


# ====================== 训练 ======================
batch_size = 64
epochs = 20
shuffle = True
for epoch in range(epochs):
    epoch_loss = global_train(model, optimizer, X_local, y_local, loss_fn, batch_size, shuffle)
    train_rmse = global_rmse_eval(model, X_local, y_local) 
    test_rmse = global_rmse_eval(model, X_test, y_test)

    if rank == 0:
        print(f"Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

print(f"done! [Rank {rank}]")