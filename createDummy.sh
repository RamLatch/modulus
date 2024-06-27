#!/bin/bash

#SBATCH --job-name=Dummy              # job name
#SBATCH --partition=dev_cpuonly                 # queue for resource allocation
#SBATCH --nodes=1                          # number of nodes to be used
#SBATCH --time=2:00:00                      # wall-clock time limit
#SBATCH --mem=243200                        # memory
#SBATCH --ntasks-per-node=1                # maximum count of tasks per node
#SBATCH --cpus-per-task=152
#SBATCH --mail-type=ALL                    # Notify user by email when certain event types occur.

BASE_DIR="/hkfs/work/workspace/scratch/ie5012-MA"
TRAIN_FILE="${BASE_DIR}/modulus/createDummy.py"

# Set up modules.
module purge                               # Unload all currently loaded modules.

source $BASE_DIR/.venvs/Modulus/bin/activate

python $TRAIN_FILE