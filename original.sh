#!/bin/bash

#SBATCH -J OriginalNet
#SBATCH --time=24:00:00
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=accelerated
#SBATCH --mem=501600mb
#SBATCH --exclusive
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/results/110824/modulus_net.out"  #results/slurm_logs/test_Dummy/mpi-%j"
#SBATCH --mail-type=ALL

module purge
module restore MA41

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
TRAIN_FILE="${BASE_DIR}/modulus/examples/weather/fcn_afno/train_era5.py"

export HYDRA_FULL_ERROR=1
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MODULUS_DISTRIBUTED_INITIALIZATION_METHOD="SLURM"
export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_P2P_DISABLE=1
#export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO
export WANDB_MODE=offline
export WANDB_START_METHOD="thread"
source $BASE_DIR/.venvs/MPI4/bin/activate
# python -c "import os;print(os.environ['CUDA_VISIBLE_DEVICES'])"
# python -c "import os,socket;from mpi4py import MPI;port = 29500;master_address = socket.gethostname(); mpi_comm = MPI.COMM_WORLD;master_address = mpi_comm.bcast(master_address, root=0);os.environ['MASTER_ADDR'] = master_address;os.environ['MASTER_PORT'] = str(port)"
# echo $MASTER_ADDR
# echo $MASTER_PORT
srun --mpi=pmix python -u $TRAIN_FILE