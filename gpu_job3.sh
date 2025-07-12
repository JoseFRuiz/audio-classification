#!/bin/bash
#SBATCH --job-name=audio-classification
#SBATCH --output=multi_gpu3.out
#SBATCH --error=mult_gpu3.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem-per-cpu=20000mb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=hpg-b200 #hpg-b200 # instead of gpu
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# module load cuda/11.1

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not available. Please install uv first."
    exit 1
fi

uv run python run_experiment_gru_lightning.py --save_dir "gru_021" --epochs 1000 --pretrained_model "gru_012" --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1
