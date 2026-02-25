#!/bin/bash
#SBATCH --job-name=trl
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --partition=P2
#SBATCH --exclude=b[05-08]
#SBATCH --output=logs/%x/%j.log
#SBATCH --error=logs/%x/%j.err


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
echo " Job ID        : $SLURM_JOB_ID"
echo " Job Name      : $SLURM_JOB_NAME"
echo " Nodes         : $SLURM_NNODES"
echo " Tasks/node    : $SLURM_NTASKS_PER_NODE"
echo " GPUs/node     : $SLURM_GPUS_ON_NODE"
echo " CPUs/Task     : $SLURM_CPUS_PER_TASK"
echo " Partition     : $SLURM_JOB_PARTITION"
echo " Mem Alloc     : $SLURM_MEM_PER_NODE"
echo " Dist. Backend : $DISTRIBUTED_BACKEND"
echo " NCCL_P2P_DISABLE : True"
echo "=============================================================="
echo " Runtime ENV"
echo "--------------------------------------------------------------"
echo " OMP_NUM_THREADS = $OMP_NUM_THREADS"
echo " MASTER_ADDR     = $MASTER_ADDR"
echo " MASTER_PORT     = $MASTER_PORT"
echo " NPROC_PER_NODE  = $NPROC_PER_NODE"
echo " SLURM_NODELIST  = $SLURM_JOB_NODELIST"
echo "=============================================================="
echo ""


# export TMPDIR="$PROJ_DIR/tmps/$SLURM_JOB_ID"
TMPDIR="$PROJ_DIR/tmps/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export WANDB_DIR="$TMPDIR/wandb"
export WANDB_CACHE_DIR="$TMPDIR/wandb_cache"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
HYDRA_ARGS="hydra.run.dir=$TMPDIR/hydra_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}"

cleanup() {
    echo "Caught signal! Cleaning up temporary directory..."
    # tmps 폴더 내부 내용 삭제 (폴더 자체를 남기려면 rm -rf $TMPDIR/*)
    rm -rf "$TMPDIR"
    exit 0
}

# SIGINT(Ctrl+C), SIGTERM(scancel 기본 신호), EXIT(정상 종료) 시 cleanup 함수 실행
trap cleanup SIGINT SIGTERM EXIT

config_name="${1:-config.yaml}"
echo "Using config: $config_name"
echo "=============================================================="
echo ""

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
