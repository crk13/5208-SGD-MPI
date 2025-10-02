# # /usr/bin/mpirun -np 12 --hostfile hostfile /home/76836/miniconda3/envs/mpi/bin/python test/main.py \
#   --n_features 17 \
#   --hidden 10 \
#   --lr 0.01 \
#   --batch_size 256 \
#   --epochs 20 \
#   --glob_interval 200 \
#   --shuffle


import numpy as np
from mpi4py import MPI
import sys, os
import time
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from source.model import Linear, Sequential, Sigmoid, ReLU, Tanh, RMSELoss, MSELoss
from source.mpi_sgd import SafeSGD, HybridSafeSGD                             
from source.data import mpi_read_data                                
from source.mpi_train import global_train, global_rmse                


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser(description="MPI Training Script")
    parser.add_argument("--n_features", type=int, default=17)
    parser.add_argument("--act", type=str, default="relu", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--hidden", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--shuffle", action="store_true") #布尔开关，不写是false，写是true
    parser.add_argument("--glob_interval", type=int, default=200,
                        help="跨机器全局同步间隔（按 batch 计）")
    args = parser.parse_args()


    # -------------------------LOAD DATA-------------------------
    n_fea = args.n_features
    comm.Barrier()
    t0 = time.time()
    training, test, ntrain, ntest = mpi_read_data("data/processed/dataall999.parquet", n_fea)
    comm.Barrier()
    t1 = time.time()
    if rank == 0:
        print(f"[Timing] Data loading time: {t1 - t0:.3f} sec")
    print(f"Loading Done! [Rank {rank}] Finished loading data. Local training samples shape: {training.shape}, test samples shape: {test.shape}")

    X_local = training[:, :-1]
    y_local = training[:, -1].reshape(-1, 1)
    X_test = test[:, :-1]
    y_test = test[:, -1].reshape(-1, 1)


    # -------------------------INITIALIZE-------------------------
    np.random.seed(42 + rank)
    ACT_MAP = {
        "relu": ReLU,
        "tanh": Tanh,
        "sigmoid": Sigmoid
    }
    act_cls = ACT_MAP[args.act]

    model = Sequential(
        Linear(n_fea, args.hidden),
        act_cls(),
        Linear(args.hidden, 1)
    )
    
    optimizer = SafeSGD(model.params, lr=args.lr, comm=comm)
    loss_fn = MSELoss()

    if rank == 0:
        print(f"Initialization Done!")


    # -------------------------TRAIN-------------------------
    for epoch in range(args.epochs):
        comm.Barrier()
        start_epoch = time.time()

        epoch_loss = global_train(model, optimizer, X_local, y_local, loss_fn, args.batch_size, args.shuffle, args.glob_interval)
        train_rmse = global_rmse(model, X_local, y_local)  
        test_rmse = global_rmse(model, X_test, y_test)

        comm.Barrier()
        end_epoch = time.time()
        if rank == 0:
            print(f"Epoch {epoch + 1}: MSELoss: {epoch_loss:.6f}, Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}, Time: {end_epoch - start_epoch:.3f} sec")

    if rank == 0:
        print("Training Done!", flush=True)

if __name__ == "__main__":
    main()
