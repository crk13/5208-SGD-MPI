# mpirun --use-hwthread-cpus -np 4 python source/datacopy.py
from mpi4py import MPI
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


def mpi_read_data(file, n_feature, max_req = 8):
    data_training, data_test = [], []
    n_training = n_test = 0

    if rank == 0:
        send_req = []
        dest = 1

        # Parquet底层是columnar storage，分成row group。
        # 每次pd.read_parquet至少会读一个或多个row group，而不是逐条。
        pf = pq.ParquetFile(file)

        print(f"[Rank {rank}] Start reading parquet file {file}, total row groups = {pf.num_row_groups}")
        for i in tqdm(range(pf.num_row_groups), disable=(rank!=0), file=sys.stderr):
            table = pf.read_row_group(i)
            # mpi4py 的 Send/Isend 只能发送连续内存的数组，比如 numpy.ndarray
            # 数值型矩阵/数组可以直接发，普通 Python 对象需要先序列化：转成 numpy array 或者用 pickle 序列化成字节流再发送
            # “普通对象”指那些不是连续内存数组的标准 Python 类型，例如：list、tuple、dict、set；str（字符串）；自定义类的实例
            # List：本身只是一个“指针数组”，每个元素实际上是指向真实对象的指针。
            # Dict / Set：哈希表实现，元素存储位置取决于哈希值和冲突处理，完全不连续。
            # NumPy ndarray：所有元素按行/列紧密排列在一块连续内存里。
            tmp = table.to_pandas().to_numpy(dtype = np.float64)

            # print(f"[Rank {rank}] Read row group {i}, shape = {tmp.shape}")

            if dest == 0:
                data_training, data_test, n_training, n_test = _divide_local_data(tmp, n_feature, data_training, data_test, n_training, n_test)
            else:
                # 直接写 tmp，MPI 不知道用什么数据类型发送（默认可能不稳定，尤其是 Fortran vs C 风格的类型）。
                # 可以直接发 Python 对象，但那是通过 pickle 序列化，速度慢，不能直接和 C/Fortran 代码互操作。
                req = comm.Isend([tmp, MPI.DOUBLE], dest, 0)
                send_req.append(req)

                if len(send_req) >= max_req:
                    MPI.Request.Waitall(send_req)
                    send_req = []

            dest = (dest + 1) % nprocs
        
        if send_req:
            MPI.Request.Waitall(send_req)
            send_req = []

        # send end signal (empty message)
        for dest in range(1, nprocs):
            req = comm.Isend([np.empty(0, dtype = np.float64), MPI.DOUBLE], dest, 1)
            send_req.append(req)

        MPI.Request.Waitall(send_req)
        # print(f"[Rank {rank}] Finished sending all data and END signals")
        # Python不要free(send_req)，因为内存管理是自动的，列表、数组和对象会在没有引用时自动回收，不像C要手动 malloc/free

    else:
        while True:
            status = MPI.Status()
            comm.Probe(0, MPI.ANY_TAG, status)
            n_items = status.Get_count(MPI.DOUBLE)
            tag = status.Get_tag()
            tmp = np.empty(n_items, dtype = np.float64)
            comm.Recv([tmp, MPI.DOUBLE], 0, tag = tag) 
            
            if tag == 1:
                break
            
            data_training, data_test, n_training, n_test = _divide_local_data(tmp, n_feature, data_training, data_test, n_training, n_test)

    # list 是 Python 的通用容器，不支持矩阵乘法，array 才支持 @、广播等操作。
    # ndarray 没有.append 方法
    return np.array(data_training, dtype=np.float64), np.array(data_test, dtype=np.float64), n_training, n_test


def _divide_local_data(tmp, n_feature, data_training, data_test, n_training, n_test):
    # MPI把数据扁平发过去,所以接收端得到的是一个一维 array，长度 = nrow × nfeature
    # +1因为有label
    tmp = tmp.reshape(-1, n_feature + 1)
    num_rows = tmp.shape[0]
    # print(f"[Rank {rank}] Splitting {num_rows} rows into train/test")

    # 生成的随机数就是在 [0, 1) 区间的浮点数。
    rd = np.random.rand(num_rows)
    for i in range(num_rows):
        if rd[i] < 0.7:
            data_training.append(tmp[i])
            n_training += 1
        else:
            data_test.append(tmp[i])
            n_test += 1
    
    return data_training, data_test, n_training, n_test

if __name__ == "__main__":
    file = "data/processed/data.parquet"
    n_fea = 19
    train, test, ntrain, ntest = mpi_read_data(file, n_fea)
    print(f"Done! [Rank {rank}] Final training shape: {train.shape}, test shape: {test.shape}")