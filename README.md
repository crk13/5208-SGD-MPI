# Distributed Single-Layer Neural Network with MPI

This project implements a **single-layer neural network** trained in a **distributed manner** using **MPI (Message Passing Interface)**.
The goal is to explore **parallel stochastic gradient descent (SGD)** strategies for large-scale data processing and efficient multi-node training.


---

## Overview

Our framework provides a clean, modular implementation of distributed training based on MPI. We use MSE as the loss function, and RMSE as the evaluation metric.


1. **MPI-based Data Loading**: Follow the DSA5208 course file 'linear_regression_simplified.c'.
   
2. **Data-Parallel SGD**: Optimize using the flatten operation. Available choices:  `sgd`, `hybrid`, `adam`.
   
3. **MPI Training**: Solve the issue where a process runs out of data.

---

## Code Structure

```
project_root/
│
├── data/
│
├── source/
│   └── __init__.py
│   └── data_cleaning.ipynb 
│   └── data.py   
│   └── model.py   
│   └── para.py
│   └── mpi_sgd.py  
│   └── mpi_train.py  
│
├── test/
│   └── main.py               # main entry point for model training
│
├── hostfile                  # MPI host configuration (list of nodes)
│
├── run.sh                    # experiment launcher script
│
└── logs                      # directory for experiment outputs
```

---

## Installation

This project requires **Python 3.8+**, **MPI (OpenMPI or MPICH)**.

### 1. Set up environment (example using Miniconda)

```bash
conda create -n mpi python=3.8
conda activate mpi
pip install torch numpy tqdm mpi4py pandas pyarrow
```

### 2. For multi-node runs

Install and configure OpenMPI:

```bash
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

Ensure `mpirun` and `hostfile` are properly configured.

---

## Running Experiments

```bash
bash run.sh
```

### Script Summary (`run.sh`)

This script automatically iterates over combinations of:

* **Batch sizes:** `512, 256, 128, 64, 32`
* **Epochs:** `[250, 230, 220, 200, 150]`
* **Activation functions:** `tanh`, `relu`, `sigmoid`
* **Optimizers (SGD variants):** `sgd`, `hybrid`, `adam`
* **Hidden units:** `60`

Each experiment runs in parallel using **16 MPI processes** specified by:

```bash
/usr/bin/mpirun -np 16 --hostfile hostfile
```

Logs are stored in:

```
logs/<SGD>/<activation>.log
```

---

## Example Command (Manual Run)

If you wish to bypass `run.sh` and run directly:

```bash
mpirun -np 8 --hostfile hostfile \
  python test/main.py \
  --n_features 21 \
  --hidden 60 \
  --lr 0.005 \
  --batch_size 256 \
  --epochs 200 \
  --act relu \
  --sgd hybrid \
  --glob_interval 200 \
  --shuffle
```