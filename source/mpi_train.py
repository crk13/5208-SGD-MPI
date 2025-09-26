import numpy as np
from mpi4py import MPI
from source.model import MSELoss

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def global_train(model, optim, X, y, lossfn=MSELoss(), batch_size=32, shuffle=True, glob_interval=1000, comm=comm):
    local_size = X.shape[0]
    max_local_size = comm.allreduce(local_size, MPI.MAX)
    local_loss = 0.0

    if shuffle:
        idx = np.arange(local_size)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, max_local_size, batch_size):
        start = i
        end = min(i + batch_size, local_size)

        if start < local_size:
            xb = X[start:end]
            yb = y[start:end]
            batch_len = xb.shape[0]

            optim.zero_grad()
            yhat = model.forward(xb)
            loss_val = lossfn.forward(yhat, yb)
            dLdy = lossfn.backward()
            model.backward(dLdy)
            if hasattr(optim, 'step_vm') and hasattr(optim, 'step_glob'):
                if (i // batch_size) % glob_interval == 0:
                    optim.step_glob(batch_len)
                else:
                    optim.step_vm(batch_len)
            else:
                optim.step(batch_len)

            local_loss += float(loss_val)

        else:
            batch_len = 0
            optim.zero_grad()

            if hasattr(optim, 'step_vm') and hasattr(optim, 'step_glob'):
                if (i // batch_size) % glob_interval == 0:
                    optim.step_glob(batch_len)
                else:
                    optim.step_vm(batch_len)
            else:
                optim.step(batch_len)
            
        if i // batch_size % 500 == 0:
            print(f"{rank}: [Batch {i//batch_size}] Loss = {loss_val: .4f}")
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # FOR MSE LOSS ONLY
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    local_loss_sum = local_loss * local_size
    global_loss_sum = comm.allreduce(local_loss_sum, MPI.SUM)
    global_size = comm.allreduce(local_size, MPI.SUM)
    return global_loss_sum / global_size


def global_rmse_scaled(model, X, y, y_max, y_min, comm=comm):
    yhat = model.forward(X)
    yhat_scale = yhat * (y_max - y_min) + y_min
    local_sq_error = np.sum((yhat_scale - y) ** 2)
    local_count = X.shape[0]

    global_sq_error = comm.allreduce(local_sq_error, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    return np.sqrt(global_sq_error / global_count)