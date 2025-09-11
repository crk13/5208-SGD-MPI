#待修改！

from mpi4py import MPI
import pandas as pd
import numpy as np

comm = MPI.COMM_WORLD
size = comm.get_size()
rank = comm.get_rank()  

def read_and_distribute_data(filename):
    if rank == 0:
        tmp = pd.read_csv(filename, chunksize = 1000)
        #选取列！！！再发送
        comm.isend(tmp, dest = i) for i in range(1, size)
    else:
        comm.irecv(data, source = 0)
        comm.wait()
        data = data.dropna(subset=['total_amount', 'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                   'passenger_count', 'trip_distance', 'RatecodeID',
                                   'PULocationID', 'DOLocationID', 'payment_type', 'extra'])
        # 特征选择
        features = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
                    'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID',
                    'payment_type', 'extra']
        target = 'total_amount'
        X = data[features]
        y = data[target]
        # 将时间特征转换为数值特征
        X['tpep_pickup_datetime'] = pd.to_datetime(X['tpep_pickup_datetime']).astype(int) / 10**9
        X['tpep_dropoff_datetime'] = pd.to_datetime(X['tpep_dropoff_datetime']).astype(int) / 10**9
        # 归一化数值特征
        X = (X - X.min()) / (X.max() - X.min())
        # 分割数据集
        split_idx = int(0.7 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train.values, y_train.values, X_test.values, y_test.values


class MPIDataLoader:
    def __init__(self, x, y, batch_size, shuffle=True, drop_last=False):
        """
        x: np.ndarray, shape (N, D)
        y: np.ndarray, shape (N,) or (N, 1)
        """
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.N = x.shape[0]
        self.indices = np.arange(self.N)
        self.cur = 0

    def __iter__(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        if self.cur >= self.N:
            raise StopIteration
        end = self.cur + self.batch_size
        if self.drop_last and end > self.N:
            raise StopIteration
        end = min(end, self.N)
        batch_idx = self.indices[self.cur: end]
        
        xb = self.x[batch_idx]
        yb = self.y[batch_idx].reshape(-1, 1)
        self.cur = end
        return xb, yb


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    batch_size = 32
    file_name = "xxx"
    X_train, y_train, X_test, y_test = read_and_distribute_data(file_name)
    train_loader = MPIDataLoader(X_train, y_train, batch_size, shuffle=True,
                                    drop_last=False)
    test_loader = MPIDataLoader(X_test, y_test, batch_size, shuffle=False,
                                    drop_last=False)
    for xb, yb in train_loader:
        print(f"Rank {rank} got batch x shape {xb.shape}, y shape {yb.shape}")
