#!/bin/bash
#SBATCH --job-name=audio-classification-7
#SBATCH --output=multi_gpu7.out
#SBATCH --error=mult_gpu7.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20gb
#SBATCH --gpus=1
#SBATCH --account=azare
#SBATCH --time=48:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Set up error handling
set -e  # Exit on any error

# Properly activate conda environment in SLURM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate audio-classification


conda deactivate