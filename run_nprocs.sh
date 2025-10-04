#!/bin/bash
set -e

PYTHON=/home/76836/miniconda3/envs/mpi/bin/python
HOSTFILE=hostfile

NPROCS_LIST=(16 15 14 13 12 11 10)

LR=0.005

BATCH_SIZES=(256)
EPOCHS_LIST=(2)
HIDDENS=(60)
ACTS=(relu)
SGDS=(sgd adam)

RESULTS_DIR=logsnpro
mkdir -p $RESULTS_DIR

for NPROCS in "${NPROCS_LIST[@]}"; do
  for SGD in "${SGDS[@]}"; do
    for HIDDEN in "${HIDDENS[@]}"; do
      for ACT in "${ACTS[@]}"; do
        LOG_DIR=$RESULTS_DIR/${SGD}
        mkdir -p $LOG_DIR
        LOG_FILE=$LOG_DIR/$ACT.log

        for i in "${!BATCH_SIZES[@]}"; do
          BATCH=${BATCH_SIZES[$i]}
          EPOCHS=${EPOCHS_LIST[$i]}

          echo "===============================" | tee -a $LOG_FILE
          echo "Start experiment" | tee -a $LOG_FILE
          echo "Config: SGD=$SGD, act=$ACT, hidden=$HIDDEN, batch_size=$BATCH, lr=$LR, epochs=$EPOCHS, NPROCS=$NPROCS" | tee -a $LOG_FILE

          /usr/bin/mpirun --oversubscribe -np $NPROCS --hostfile $HOSTFILE \
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
done
