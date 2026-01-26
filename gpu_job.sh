#!/bin/bash
#SBATCH --job-name=audio-classification
#SBATCH --output=multi_gpu.out
#SBATCH --error=mult_gpu.err
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

python run_experiment_dino_lightning.py \
  --save_dir "supervised_only_002" \
  --supervised_only \
  --finetune_epochs 500 \
  --eval_interval 10 \
  --log_interval 10 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --gradient_clip_val 10.0 \
  --batch_size 100 \
  --use_gpu \
  --test_size 0.1 \
  --dropout 0.1 \
  --num_workers 4 \
  --conv_channels 64 128 256 \
  --conv_kernel_size 3 \
  --conv_stride 1 \
  --dataset_type "complete" \
  --bptt_length 60 \
  --labeled_ratio 0.1 \
  --init_weights xavier

conda deactivate