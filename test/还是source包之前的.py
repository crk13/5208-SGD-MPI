# import numpy as np
# from mpi4py import MPI
# import argparse

# from source.data_process import read_and_distribute_data, MPIDataLoader
# from source.model import Linear, ReLU, Tanh, MSELoss, Module, Para
# from source.optimizer import SGD

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# file_name = "data/Concrete_Data.csv"
# dataloader = MPIDataLoader()
# loss_fn = MSELoss()


# main.py
import argparse
import numpy as np
from mpi4py import MPI
from source.model import MSELoss
from source.para import Para
from source.mpi_sgd import SGD
from source.data_process import DataLoader  # 你自己实现的 DataLoader
from source.model import Linear, ReLU, Tanh  # 为了 MyMLP class

# -----------------------
# 封装模型 class
# -----------------------
class MyMLP:
    def __init__(self, input_dim, hidden_dim, output_dim, activation="relu"):
        self.W1 = Para(np.random.randn(input_dim, hidden_dim) * 0.01)
        self.b1 = Para(np.zeros((1, hidden_dim)))
        self.linear1 = Linear(self.W1, self.b1)

        if activation.lower() == "relu":
            self.act = ReLU()
        elif activation.lower() == "tanh":
            self.act = Tanh()
        else:
            raise ValueError("Unsupported activation")

        self.W2 = Para(np.random.randn(hidden_dim, output_dim) * 0.01)
        self.b2 = Para(np.zeros((1, output_dim)))
        self.linear2 = Linear(self.W2, self.b2)

        self.params = [self.W1, self.b1, self.W2, self.b2]

    def forward(self, x):
        out = self.linear1.forward(x)
        out = self.act.forward(out)
        out = self.linear2.forward(out)
        return out

    def backward(self, dLdy):
        grad = self.linear2.backward(dLdy)
        grad = self.act.backward(grad)
        grad = self.linear1.backward(grad)
        return grad

# -----------------------
# 训练和评估函数
# -----------------------
def train_loop(model, optimizer, dataloader, loss_fn, epochs):
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            # forward
            yhat = model.forward(x_batch)
            loss = loss_fn.forward(yhat, y_batch)
            epoch_loss += loss

            # backward
            dLdy = loss_fn.backward()
            model.backward(dLdy)

            # MPI 参数更新
            optimizer.step()

        if dataloader.rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss={epoch_loss/len(dataloader)}")

def evaluate(model, dataloader, loss_fn):
    total_loss = 0.0
    for x_batch, y_batch in dataloader:
        yhat = model.forward(x_batch)
        loss = loss_fn.forward(yhat, y_batch)
        total_loss += loss

    total_loss_sync = np.array([total_loss], dtype=np.float64)
    dataloader.comm.Allreduce(MPI.IN_PLACE, total_loss_sync, op=MPI.SUM)
    return total_loss_sync[0] / dataloader.comm.Get_size()

# -----------------------
# 主程序入口
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MPI Distributed Training")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="data/nytaxi2022.csv")
    parser.add_argument("--activation", type=str, default="relu")
    args = parser.parse_args()

    # MPI 初始化
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 数据加载
    dataloader = DataLoader(args.data_path, batch_size=args.batch_size, comm=comm)
    input_dim = dataloader.input_dim
    output_dim = dataloader.output_dim

    # 构建模型和优化器
    model = MyMLP(input_dim, args.hidden_dim, output_dim, activation=args.activation)
    optimizer = SGD(params=model.params, lr=args.lr, comm=comm)

    # 损失函数
    loss_fn = MSELoss()

    # 训练
    train_loop(model, optimizer, dataloader.train_loader, loss_fn, args.epochs)

    # 评估训练集
    train_loss = evaluate(model, dataloader.train_loader, loss_fn)
    if rank == 0:
        print(f"Training Loss: {train_loss}")

    # 评估测试集
    test_loss = evaluate(model, dataloader.test_loader, loss_fn)
    if rank == 0:
        print(f"Test Loss: {test_loss}")
