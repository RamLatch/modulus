#!/usr/bin/env bash 

#SBATCH --time=0:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH -J compare
#SBATCH --output="/hkfs/work/workspace/scratch/ie5012-MA/debug/slurm_logs/compare"
#SBATCH -p dev_accelerated
#SBATCH --mem=1600mb
#SBATCH --mail-type=END

ml purge

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
TRAIN_FILE="${BASE_DIR}/modulus/compare.py"

source $BASE_DIR/.venvs/Modulus/bin/activate
python $TRAIN_FILE