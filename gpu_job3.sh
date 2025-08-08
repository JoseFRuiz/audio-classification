#!/bin/bash
#SBATCH --job-name=audio-classification-3
#SBATCH --output=multi_gpu3.out
#SBATCH --error=mult_gpu3.err
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

python run_experiment_gru_lightning.py --save_dir "gru_033" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1 --num_workers 1

conda deactivate