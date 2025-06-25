#!/bin/bash

### SBATCH options
#SBATCH --account=hw_nresearch_snoise
#SBATCH --job-name=send_recv
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=8


NODES=${SLURM_JOB_NUM_NODES:-1}
echo "Allocated nodes: $SLURM_JOB_NUM_NODES"
# echo "NODES = $NODES"

export ONE_LOGGER_JOB_CATEGORY=test

# Configurables
IMAGE=/lustre/fsw/portfolios/hw/users/sshrestha/nvidia+nemo+25.04.sqsh

CONTAINER_WORKDIR=/mounted_ws

# original workspace mount
WORK_MOUNT=/home/sshrestha/workspace/comms:/mounted_ws

# mount your home directory
HOME_MOUNT=${HOME}:${HOME}

# mount an additional Lustre filesystem (replace with actual path)
EXTRA_FS_MOUNT=/lustre/fsw/portfolios/hw/users/sshrestha/:/lustre/fsw/portfolios/hw/users/sshrestha/
CONTAINER_MOUNTS="${WORK_MOUNT},${HOME_MOUNT},${EXTRA_FS_MOUNT}"

RESULTS_DIR=/home/sshrestha/workspace/comms/sbatch_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "${RESULTS_DIR}"

SCRIPT="/home/sshrestha/workspace/Megatron-LM/tensor_send_recv.py"
ARGS="--batch_size 128 --hidden_dim 8192 --warmup_iters 10 --timing_iters 10 --verify_data"

srun --mpi=pmix                                                            \
    --nodes=$NODES                                                        \
    --ntasks-per-node=1                                                   \
    --container-image=$IMAGE                                              \
    --container-mounts=$CONTAINER_MOUNTS                                  \
    --container-workdir=$CONTAINER_WORKDIR                                \
    --container-writable                                                  \
    --no-container-mount-home                                             \
    --output="${RESULTS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"      \
    --error="${RESULTS_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"      \
    bash -c "python $SCRIPT $ARGS"
