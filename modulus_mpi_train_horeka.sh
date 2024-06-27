#!/usr/bin/env bash 

#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1

#SBATCH -J modulus_Test
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/results/slurm_logs/test_modulus/slurm-%j"
#SBATCH -p dev_accelerated
#SBATCH --mem=501600mb
#SBATCH --exclusive
#SBATCH --mail-type=ALL

ml purge
ml load compiler/intel/2023.1.0
ml load mpi/openmpi/4.1

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
TRAIN_FILE="${BASE_DIR}/modulus/examples/weather/fcn_afno/train_era5.py"

export HYDRA_FULL_ERROR=1
# export HDF5_USE_FILE_LOCKING=FALSE
# export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355

# SRUN_PARAMS=(
#   --mpi="pmi2"
#   --gpus-per-task=1
#   --gpu-bind="closest"
#   --label
# )

source $BASE_DIR/.venvs/Modulus/bin/activate
# srun -u --mpi=pmi2 bash -c " 
#   python $TRAIN_FILE"
mpirun python $TRAIN_FILE