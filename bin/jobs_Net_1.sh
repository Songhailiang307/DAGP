#!/bin/bash
#SBATCH --job-name=Net_1
#SBATCH --output=Net_1_%A_%a.out
#SBATCH --error=Net_1_%A_%a.err
#SBATCH --array=0-7             # Array jobs: n tasks
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=1              # Number of tasks per node
#SBATCH --cpus-per-task=56       # Number of CPUs per task
#SBATCH --mem=480GB               # Memory per node
#SBATCH --partition=nnode     # Partition name


module load python/3.8.3
module load conda/3-2020.07

# Run the Python script using MPI
python ./bin/Net_1_slurm.py