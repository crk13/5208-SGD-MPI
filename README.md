# 5208-SGD-MPI
Course project for Scalable Distributed Computing (5208)

本仓库包含课程 **5208 Scalable Distributed Computing for Data Science** 的小组作业代码与报告。  
任务要求：使用 **MPI 并行随机梯度下降 (SGD)** 训练一隐层神经网络，预测出租车数据集 (`nytaxi2022.csv`) 的 `total_amount`。

---

## 👥 小组成员
- 李同学（负责 MPI 框架与并行实现）
- 刘同学（负责模型实现与算法推导）
- 王同学（负责数据处理与实验运行）

---

## 📂 仓库结构
5208-SGD-MPI/
│
├── src/              # 源代码（Python 或 C，取决于实现）
├── data/             # 数据处理脚本（不包含原始大数据集）
├── experiments/      # 实验结果与日志（RMSE、曲线、性能数据）
├── report/           # 报告初稿与最终 PDF
├── README.md         # 使用说明（本文件）
└── .gitignore        # 忽略规则

---

## ⚙️ 环境依赖
- Python 3.9+
- 常用库：
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `mpi4py`
- MPI 环境（Open MPI 或 MPICH，版本需一致）

安装依赖：
```bash
pip install -r requirements.txt
```

## 运行方法

1. 数据预处理


将原始数据放入 data/raw/ 文件夹，运行：

```bash
python data/preprocess.py
```
2. 并行训练

使用 mpirun 运行主程序，例如：

```
mpirun -np 4 python src/train.py --batch-size 128 --hidden 64 --activation relu
```

3. 参数说明
	•	--batch-size：小批量大小 (M)
	•	--hidden：隐藏层神经元数量 (n)
	•	--activation：激活函数（relu / tanh / sigmoid）
	•	--lr：学习率
	•	--epochs：最大迭代轮数

实验结果
	•	训练曲线 (Loss vs Iteration)
	•	训练/测试集 RMSE
	•	不同进程数的运行时间与加速比

（详细结果见 experiments/ 文件夹）

## 报告

完整的项目报告位于 report/ 文件夹，最终提交 PDF 名为：

`5208_Group_Report.pdf`

## 注意事项
	•	原始 CSV 数据过大，不上传至仓库，只提供处理脚本。
	•	请确保三人使用 相同版本的 MPI，避免兼容性问题。
	•	开发时请在各自的分支完成修改，再合并到 main。