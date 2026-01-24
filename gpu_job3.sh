#!/bin/bash
#SBATCH --job-name=audio-classification-3
#SBATCH --output=multi_gpu3.out
#SBATCH --error=mult_gpu3.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20gb
#SBATCH --gpus=1
#SBATCH --account=azare
#SBATCH --time=72:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Set up error handling
set -e  # Exit on any error

# Properly activate conda environment in SLURM
source ~/anaconda3/etc/profile.d/conda.sh
conda activate audio-classification

python run_experiment_gru_lightning.py --save_dir "mel_gru_attn_001" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "mel" --model_type "gru" --use_attention --attention_heads 8 --n_mels 128 --n_fft 2048 --hop_length 512

conda deactivate