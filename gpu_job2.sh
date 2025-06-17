#!/bin/bash
#SBATCH --job-name=audio-classification
#SBATCH --output=multi_gpu2.out
#SBATCH --error=mult_gpu2.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem-per-cpu=20000mb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

module load cuda/11.1

source activate audio-classification

python run_experiment_gru_lightning.py --save_dir "gru_020" --epochs 1000 --pretrained_model "gru_011" --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0

conda deactivate
