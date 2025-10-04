import numpy as np
from mpi4py import MPI
from source.model import MSELoss
from tqdm import tqdm
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def global_train(model, optim, X, y, lossfn=None, batch_size=32, shuffle=True, glob_interval=1000, comm=None):
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if lossfn is None:
        from source.model import RMSELoss
        lossfn = RMSELoss()
    
    local_size = X.shape[0]
    max_local_size = comm.allreduce(local_size, MPI.MAX)
    local_sse = 0.0
    local_n = 0

    if shuffle and local_size > 0:
        idx = np.arange(local_size)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in tqdm(range(0, max_local_size, batch_size), disable=(rank!=0), file=sys.stderr):
        start = i
        end = min(i + batch_size, local_size)

        if start < local_size:
            xb = X[start:end]
            yb = y[start:end]
            batch_len = xb.shape[0]

            optim.zero_grad()
            yhat = model.forward(xb)
            err = yhat - yb
            local_sse += float((err*err).sum())
            local_n += batch_len

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
            
        # if i // batch_size % 10000 == 0:
        #     print(f"{rank}: [Batch {i//batch_size}] Loss = {loss_val: .4f}")
    
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # # FOR RMSE LOSS ONLY
    # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # total_sse = comm.allreduce(local_sse, MPI.SUM)
    # total_n = comm.allreduce(local_n, MPI.SUM)
    # return np.sqrt(total_sse / max(total_n, 1))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # FOR MSE LOSS ONLY
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    total_sse = comm.allreduce(local_sse, MPI.SUM)
    total_n = comm.allreduce(local_n, MPI.SUM)
    return 0.5 * total_sse / max(total_n, 1)


def global_rmse(model, X, y, comm=comm):
    yhat = model.forward(X)
    local_sq_error = np.sum((yhat - y) ** 2)
    local_count = X.shape[0]

    global_sq_error = comm.allreduce(local_sq_error, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    return np.sqrt(global_sq_error / global_count)