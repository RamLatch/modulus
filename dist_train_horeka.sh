#!/usr/bin/env bash 

#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2

#SBATCH -J m_dist_T
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/results/slurm_logs/test_080724/mpi-tw0-%j.out"  #results/slurm_logs/test_Dummy/mpi-%j"
#SBATCH -p dev_accelerated
#SBATCH --mem=501600mb
#SBATCH --exclusive
#SBATCH --mail-type=ALL

ml purge
ml load compiler/intel/2023.1.0
ml load mpi/openmpi/4.1

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
TRAIN_FILE="${BASE_DIR}/modulus/examples/weather/fcn_afno/train_era5_MPI.py"

# export RANK=0
# export WORLD_SIZE=1
# export MASTER_PORT=$RANDOM
export HYDRA_FULL_ERROR=1
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)

# SRUN_PARAMS=(
#   --mpi="pmi2"
#   --gpus-per-task=1
#   --gpu-bind="closest"
#   --label
# )

source $BASE_DIR/.venvs/Modulus/bin/activate
# srun -u --mpi=pmi2 bash -c " 
#   python $TRAIN_FILE"
mpirun --bind-to core --map-by core -report-bindings -bind-to none --mca mpi_warn_on_fork 0 python -u $TRAIN_FILE