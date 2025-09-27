#!/bin/bash
set -e

# 参数设置
EPOCHS=160
HIDDEN=30
LR=0.005
BATCH_SIZES=(256 128 64 32 16)
NPROCS=12
HOSTFILE=hostfile
PYTHON=/home/76836/miniconda3/envs/mpi/bin/python

RESULTS_DIR=results
mkdir -p $RESULTS_DIR

for BATCH in "${BATCH_SIZES[@]}"; do
    EXP_DIR=$RESULTS_DIR/ReLU/batch_${BATCH}
    mkdir -p $EXP_DIR
    
    echo "===============================" | tee -a $EXP_DIR/summary.log
    echo "Start experiment at $(date)" | tee -a $EXP_DIR/summary.log
    echo "Config: batch_size=$BATCH, hidden=$HIDDEN, lr=$LR, epochs=$EPOCHS" | tee -a $EXP_DIR/summary.log
    
    # 直接运行 MPI，Python 脚本内部处理日志
    /usr/bin/mpirun -np $NPROCS --hostfile $HOSTFILE \
        $PYTHON test/main.py \
        --n_features 17 \
        --hidden $HIDDEN \
        --lr $LR \
        --batch_size $BATCH \
        --epochs $EPOCHS \
        --shuffle \
        --glob_interval 200
    
    echo "" | tee -a $EXP_DIR/summary.log
done
