#!/usr/bin/env bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:01:00

#SBATCH -J DummyTest
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/results/slurm_logs/test_Dummy/slurm-%j"
#SBATCH -p dev_cpuonly
#SBATCH --mem=1600mb
#SBATCH --mail-type=ALL

ml purge

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
BATCH_FILE_SGL="${BASE_DIR}/modulus/train_horeka.sh"
BATCH_FILE_MPI="${BASE_DIR}/modulus/mpi_train_horeka.sh"

sbatch $BATCH_FILE_SGL
sbatch $BATCH_FILE_MPI
