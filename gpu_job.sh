#!/bin/bash
#SBATCH --job-name=audio-classification
#SBATCH --output=multi_gpu.out
#SBATCH --error=mult_gpu.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
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

python run_experiment_asit_lightning.py \
  --save_dir "asit_supervised_009" \
  --supervised_only \
  --finetune_epochs 700 \
  --vit_model base \
  --labeled_ratio 0.1 \
  --dataset_type "complete" \
  --batch_size 32 \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --dropout 0.1 \
  --eval_interval 1 \
  --gradient_accumulation_steps 2 \
  --use_gpu

conda deactivate