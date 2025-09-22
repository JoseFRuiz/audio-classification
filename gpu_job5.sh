#!/bin/bash
#SBATCH --job-name=audio-classification-5
#SBATCH --output=multi_gpu5.out
#SBATCH --error=mult_gpu5.err
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

python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_016" --epochs 500 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4 --use_bidirectional --use_attention --bptt_length 60 --attention_heads 8 --gradient_accumulation_steps 2
python run_birdclef_experiment.py --audio_dir ./birdclef_data/train_audio --csv_path ./birdclef_data/birdclef_2023_dataset.csv --save_dir "birdclef_017" --epochs 500 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_asymmetric" --wu_weight 0.5 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4 --use_bidirectional --use_attention --bptt_length 60 --attention_heads 8 --gradient_accumulation_steps 2

conda deactivate