#!/bin/bash
#SBATCH --job-name=trl_v3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/%x/%j.log
#SBATCH --error=logs/%x/%j.err


# 정해야 하는 것
# --partition
# --nodes
# --gres=gpu:
# --mem
# --time

# (필요한 경우) 정해야 하는 것
# --nodelist
# --exclude


PROJ_DIR="/home/s1/jongmin/core/training/trl_SFTTrainer"
cd $PROJ_DIR


export NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((10000 + (SLURM_JOB_ID % 50000)))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DISTRIBUTED_BACKEND=nccl
export NCCL_P2P_DISABLE=1


echo "=============================================================="
echo " SLURM JOB INFORMATION"
echo "--------------------------------------------------------------"
echo " Job ID           : $SLURM_JOB_ID"
echo " Job Name         : $SLURM_JOB_NAME"
echo " Nodes            : $SLURM_NNODES"
echo " Tasks/node       : $SLURM_NTASKS_PER_NODE"
echo " GPUs/node        : $SLURM_GPUS_ON_NODE"
echo " CPUs/Task        : $SLURM_CPUS_PER_TASK"
echo " Partition        : $SLURM_JOB_PARTITION"
echo " SLURM_NODELIST   : $SLURM_JOB_NODELIST"
echo " Mem Alloc        : $SLURM_MEM_PER_NODE"
echo "=============================================================="
echo " Runtime ENV"
echo "--------------------------------------------------------------"
echo " OMP_NUM_THREADS  : $OMP_NUM_THREADS"
echo " MASTER_ADDR      : $MASTER_ADDR"
echo " MASTER_PORT      : $MASTER_PORT"
echo " NPROC_PER_NODE   : $NPROC_PER_NODE"
echo " Dist. Backend    : $DISTRIBUTED_BACKEND"
echo " NCCL_P2P_DISABLE : True"
echo "=============================================================="
echo ""

mkdir -p "$PROJ_DIR/tmps"
export WANDB_DIR="$PROJ_DIR/tmps/wandb"
export WANDB_CACHE_DIR="$PROJ_DIR/tmps/wandb_cache"
export TORCHINDUCTOR_CACHE_DIR="$PROJ_DIR/tmps/torchinductor"
export TRITON_CACHE_DIR="$PROJ_DIR/tmps/triton"

cleanup() {
    echo "Caught signal! Cleaning up temporary directory..."
    # tmps 폴더 내부 내용 삭제 (폴더 자체를 남기려면 rm -rf $TMPDIR/*)
    rm -rf "$PROJ_DIR/tmps"
    exit 0
}

# SIGINT(Ctrl+C), SIGTERM(scancel 기본 신호), EXIT(정상 종료) 시 cleanup 함수 실행
trap cleanup SIGINT SIGTERM EXIT


config_name="${1:-config.yaml}"
echo "Using config: $config_name"
echo "=============================================================="
echo ""
HYDRA_ARGS=""

srun --label uv run torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc-per-node=$NPROC_PER_NODE \
    --node-rank=$SLURM_NODEID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv-id=$SLURM_JOB_ID \
    main.py \
    --config-name $config_name \
    $HYDRA_ARGS  # Hydra 경로 설정 인자