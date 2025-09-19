from src import data_preprocessing_wqc
from src import mpi

if __name__ == "__main__":
    # Step 1: 数据预处理
    print("=== Running preprocessing ===")
    # 可以直接调用预处理脚本的入口函数（假设它定义在 if __name__ == "__main__": 中）
    data_preprocessing_wqc.main()

    # Step 2: 分布式训练
    print("=== Running MPI training ===")
    mpi.main()