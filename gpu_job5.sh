#!/bin/bash
#SBATCH --job-name=audio-classification-5
#SBATCH --output=multi_gpu5.out
#SBATCH --error=mult_gpu5.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --gpus=1
#SBATCH --account=azare
#SBATCH --time=48:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Set up error handling
set -e  # Exit on any error

source activate audio-classification

python run_experiment_gru_lightning.py --save_dir "gru_043" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4

conda deactivate