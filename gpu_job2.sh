#!/bin/bash
#SBATCH --job-name=audio-classification-2
#SBATCH --output=multi_gpu2.out
#SBATCH --error=mult_gpu2.err
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

# Properly activate conda environment in SLURM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate audio-classification

python run_experiment_gru_lightning.py --save_dir "wav2vec_008" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "wav2vec" --window_size 1024 --hop_size 512

conda deactivate