#!/bin/bash
set -e

N_fea=19
ACT="tanh"
SGD="hybrid"
HIDDEN=60
LR=0.005
BATCH_SIZES=(512)
EPOCHS=160
Glob_interval=200

NPROCS=16
HOSTFILE=hostfile
PYTHON=/home/76836/miniconda3/envs/mpi/bin/python

   
NET_IFACE=$(ip route | awk '/default/ {print $5}')  # ens4

echo "[INFO] Simulating WLAN: 30ms delay, 30Mbps bandwidth on all hosts"
while read HOST; do
    # 去掉行末非法字符（如 \r、空格）
    HOST=$(echo "$HOST" | tr -d '\r' | awk '{print $1}')

    # 跳过空行
    [ -z "$HOST" ] && continue

    # 判断是否为 localhost，直接本地执行
    if echo "$HOST" | grep -q "localhost"; then
        sudo tc qdisc add dev "$NET_IFACE" root netem delay 50ms rate 20mbit
    else
        ssh "$HOST" "sudo tc qdisc add dev $NET_IFACE root netem delay 50ms rate 20mbit"
    fi
done < "$HOSTFILE"




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
        --n_features $N_fea \
        --hidden $HIDDEN \
        --lr $LR \
        --batch_size $BATCH \
        --epochs $EPOCHS \
        --act $ACT \
        --sgd $SGD \
        --glob_interval $Glob_interval \
        --shuffle
    
done



echo "[INFO] Clearing network limits on all hosts"
while read HOST; do
    # 去掉行末非法字符（如 \r、空格）
    HOST=$(echo "$HOST" | tr -d '\r' | awk '{print $1}')

    # 跳过空行
    [ -z "$HOST" ] && continue

    # 判断是否为 localhost，直接本地执行
    if echo "$HOST" | grep -q "localhost"; then
        sudo tc qdisc del dev ens4 root || true
    else
        ssh $HOST "sudo tc qdisc del dev ens4 root || true"
    fi
done < "$HOSTFILE"

