#mpirun -np 1 python test/demo.py

import numpy as np
from mpi4py import MPI
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.model import Linear, Sequential, ReLU, Tanh, Sigmoid, RMSELoss
from source.para import MPITensor
from source.mpi_sgd import SGD, HybridSafeSGD
from source.mpi_train import global_train, global_rmse 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_features = 10
hidden = 10
np.random.seed(42 + rank)
n_samples = 1000
lr = 0.1
epochs = 10
batch_size = 50

X = np.random.randn(n_samples, n_features)
y = np.random.randn(n_samples, 1)
# 每个进程划分一部分数据
local_size = n_samples // size
start = rank * local_size
end = (rank + 1) * local_size
X_local = X[start:end]
y_local = y[start:end]

model = Sequential(Linear(n_features, hidden), 
                   Tanh(), 
                   Linear(hidden, 1))
optim = HybridSafeSGD(model.params, lr, comm)
loss_fn = RMSELoss()

for epoch in range(4):
    epoch_loss = global_train(model, optim, X_local, y_local, loss_fn, 64, True, 50)
    train_rmse = global_rmse(model, X_local, y_local)
    if rank == 0:
        print(f"Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, Train RMSE: {train_rmse:.4f}")

# # 训练
# for epoch in range(epochs):
#     epoch_loss = 0.0
#     for i in range(0, local_size, batch_size):
#         xb = X_local[i:i+batch_size]
#         yb = y_local[i:i+batch_size]

#         optimizer.zero_grad()
#         yhat = model.forward(xb)
#         loss = loss_fn.forward(yhat, yb)
#         dLdy = loss_fn.backward()
#         model.backward(dLdy)
#         optimizer.step()
#         epoch_loss += loss

#     # 同步 loss
#     loss_arr = np.array([epoch_loss], dtype=np.float64)
#     comm.Allreduce(MPI.IN_PLACE, loss_arr, op=MPI.SUM)
#     if rank == 0:
#         print(f"Epoch {epoch+1}, Loss={loss_arr[0]/size:.4f}")

print(f"done! [Rank {rank}]")