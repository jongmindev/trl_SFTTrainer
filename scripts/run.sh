#!/bin/bash


PARTITION=$1
NNODES=$2

if [ -z "$PARTITION" ]; then
    echo "Usage: bash run.sh PARTITION NNODES"
    exit 1
fi

if [ -z "$NNODES" ]; then
    echo "Usage: bash run.sh PARTITION NNODES"
    exit 1
fi

# PARTITION 이 P2, thunder1 인 경우
if [ "$PARTITION" == "P2" ] || ["$PARTITION" == "thunder1" ]; then
    TIME="12:00:00"
    GPUS=4
    MEM=400GB
elif [ "$PARTITION" == "EA" ]; then
    TIME=infinite
    GPUS=6
    MEM=200GB
else
    echo "Invalid partition: $PARTITION"
    exit 1
fi

echo "Partition : $PARTITION"
echo "NNODES    : $NNODES"
echo "GPUS      : $GPUS"
echo "MEM       : $MEM"
echo "TIME      : $TIME"

sbatch \
    --partition=$PARTITION \
    --nodes=$NNODES \
    --gres=gpu:$GPUS \
    --mem=$MEM \
    --time=$TIME \
    slurm.sh
