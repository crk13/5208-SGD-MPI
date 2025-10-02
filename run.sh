#!/bin/bash
set -e

EPOCHS=160
ACT="relu"
HIDDEN=20
LR=0.005
BATCH_SIZES=(128)
NPROCS=12
HOSTFILE=hostfile
PYTHON=/home/76836/miniconda3/envs/mpi/bin/python

MODE="WLAN"   
NET_IFACE=$(ip route | awk '/default/ {print $5}')
if [ "$MODE" = "WLAN" ]; then
    echo "Simulating WLAN: 50ms delay, 20Mbps bandwidth"
    sudo tc qdisc add dev $NET_IFACE root netem delay 50ms rate 20mbit
fi

RESULTS_DIR=resultsss
mkdir -p $RESULTS_DIR

for BATCH in "${BATCH_SIZES[@]}"; do
    EXP_DIR=$RESULTS_DIR/${ACT}
    mkdir -p $EXP_DIR
    
    echo "===============================" | tee -a $EXP_DIR/summary.log
    echo "Start experiment" | tee -a $EXP_DIR/summary.log
    echo "Config: batch_size=$BATCH, hidden=$HIDDEN, lr=$LR, epochs=$EPOCHS" | tee -a $EXP_DIR/summary.log
    
    /usr/bin/mpirun -np $NPROCS --hostfile $HOSTFILE \
        $PYTHON test/main.py \
        --n_features 19 \
        --act $ACT \
        --hidden $HIDDEN \
        --lr $LR \
        --batch_size $BATCH \
        --epochs $EPOCHS \
        --shuffle \
        --glob_interval 200
    
    echo "" | tee -a $EXP_DIR/summary.log
done

if [ "$MODE" = "WLAN" ]; then
    echo "[INFO] Clearing network limits"
    sudo tc qdisc del dev $NET_IFACE root
fi
