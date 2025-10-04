#!/bin/bash
set -e

HOSTFILE=hostfile
MODE="WLAN"   
NET_IFACE=$(ip route | awk '/default/ {print $5}')  # ens4

if [ "$MODE" = "WLAN" ]; then
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
fi