#!/bin/bash
set -e

PYTHON=/home/76836/miniconda3/envs/mpi/bin/python
HOSTFILE=hostfile
NPROCS=16

LR=0.005

BATCH_SIZES=(512 256 128 64 32)
EPOCHS_LIST=(350 300 250 200 150)  
HIDDENS=(80 60 40 20)
ACTS=(relu tanh sigmoid)
SGDS=(adam sgd)

RESULTS_DIR=logs
mkdir -p $RESULTS_DIR

for SGD in "${SGDS[@]}"; do
  for ACT in "${ACTS[@]}"; do
    for HIDDEN in "${HIDDENS[@]}"; do
      LOG_DIR=$RESULTS_DIR/$SGD
      mkdir -p $LOG_DIR
      LOG_FILE=$LOG_DIR/$ACT.log

      for i in "${!BATCH_SIZES[@]}"; do
        BATCH=${BATCH_SIZES[$i]}
        EPOCHS=${EPOCHS_LIST[$i]}

        echo "===============================" | tee -a $LOG_FILE
        echo "Start experiment" | tee -a $LOG_FILE
        echo "Config: SGD=$SGD, act=$ACT, hidden=$HIDDEN, batch_size=$BATCH, lr=$LR, epochs=$EPOCHS" | tee -a $LOG_FILE

        /usr/bin/mpirun -np $NPROCS --hostfile $HOSTFILE \
          $PYTHON test/main.py \
          --n_features 21 \
          --hidden $HIDDEN \
          --lr $LR \
          --batch_size $BATCH \
          --epochs $EPOCHS \
          --act $ACT \
          --sgd $SGD \
          --glob_interval 200 \
          --shuffle

      done
    done
  done
done
