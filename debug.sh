#!/usr/bin/env bash 

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:01:00

#SBATCH -J Debug
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/results/NONE"
#SBATCH -p dev_cpuonly
#SBATCH --mem=1600mb

ml purge

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
BATCH_FILE_SGL="${BASE_DIR}/modulus/train_horeka.sh"
BATCH_FILE_MPI="${BASE_DIR}/modulus/mpi_train_horeka.sh"
BATCH_FILE_COM="${BASE_DIR}/modulus/compare.sh"

jobID1=$(sbatch $BATCH_FILE_SGL 2>&1 | sed 's/[S,a-z]* //g')
jobID2=$(sbatch --dependency=afternotok:${jobID1} $BATCH_FILE_MPI 2>&1 | sed 's/[S,a-z]* //g')
sbatch --dependency=afternotok:${jobID1}:${jobID2} $BATCH_FILE_COM
