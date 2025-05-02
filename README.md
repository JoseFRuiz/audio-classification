# audio-classification

## Environment Setup

1. Create a new conda environment:
```bash
conda create -n audio-classification python=3.8
conda activate audio-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running Experiments

To train the model:
```bash
python run_experiment_gru.py --save_dir "results" --epochs 1000 --eval_interval 10 --lr 1e-3 --batch_size 32 --use_gpu
```

For GPU cluster environments, you can use the provided batch script:
```bash
sbatch gpu_job.sh
```