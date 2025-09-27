# mpirun --use-hwthread-cpus -np 4 python test/test.py
# /usr/bin/mpirun -np 12 --hostfile hostfile /home/76836/miniconda3/envs/mpi/bin/python test/test.py


import numpy as np
from mpi4py import MPI
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.model import Linear, Sequential, ReLU, Tanh, Sigmoid, MSELoss, RMSELoss
from source.mpi_sgd import SafeSGD, HybridSafeSGD
from source.data import mpi_read_data
from source.mpi_train import global_train, global_rmse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ====================== Load Data ======================
n_fea = 17
training, test, ntrain, ntest = mpi_read_data("data/processed/dataall.parquet", n_fea)
print(f"Loading Done! [Rank {rank}] Finished loading data. Local training samples shape: {training.shape}, test samples shape: {test.shape}")

X_local = training[:, :-1]
y_local = training[:, -1].reshape(-1, 1)
X_test = test[:, :-1]
y_test = test[:, -1].reshape(-1, 1)



# ====================== Initialize ======================
hidden = 16
lr = 0.01
np.random.seed(42 + rank)


model = Sequential(
    Linear(n_fea, hidden), 
    Sigmoid(),
    Linear(hidden, 1)                  
)
optimizer = SafeSGD(model.params, lr, comm)
loss_fn = RMSELoss()


# ====================== Train & Eval ======================
batch_size = 64
epochs = 20
shuffle = True
glob_interval=200
for epoch in range(epochs):

    epoch_loss = global_train(model, optimizer, X_local, y_local, loss_fn, batch_size, shuffle, glob_interval)
    train_rmse = global_rmse(model, X_local, y_local)
    test_rmse = global_rmse(model, X_test, y_test)

    if rank == 0:
        print(f"Epoch {epoch + 1}: RMSE Loss: {epoch_loss:.4f}, Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

print(f"done! [Rank {rank}]")