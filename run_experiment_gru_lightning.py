# python run_experiment_gru_lightning.py --save_dir "gru_002" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu
# python run_experiment_gru_lightning.py --save_dir "gru_003" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu
# python run_experiment_gru_lightning.py --save_dir "gru_004" --epochs 1000 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_005" --epochs 100 --eval_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_006" --epochs 1000 --eval_interval 100 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --pretrained_model "gru_005"
# python run_experiment_gru_lightning.py --save_dir "gru_007" --epochs 1000 --eval_interval 10 --lr 1e-4 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_008" --epochs 10000 --eval_interval 100 --lr 1e-4 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --pretrained_model "gru_007"
# python run_experiment_gru_lightning.py --save_dir "gru_009" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_010" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce"
# python run_experiment_gru_lightning.py --save_dir "gru_011" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0
# python run_experiment_gru_lightning.py --save_dir "gru_012" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_013" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce"
# python run_experiment_gru_lightning.py --save_dir "gru_014" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0
# python run_experiment_gru_lightning.py --save_dir "gru_015" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_016" --epochs 1000 --pretrained_model "gru_013" --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce"
# python run_experiment_gru_lightning.py --save_dir "gru_017" --epochs 1000 --pretrained_model "gru_014" --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0
# python run_experiment_gru_lightning.py --save_dir "gru_018" --epochs 1000 --pretrained_model "gru_015" --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_019" --epochs 1000 --pretrained_model "gru_010" --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce"
# python run_experiment_gru_lightning.py --save_dir "gru_020" --epochs 1000 --pretrained_model "gru_011" --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0
# python run_experiment_gru_lightning.py --save_dir "gru_021" --epochs 1000 --pretrained_model "gru_012" --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1
# python run_experiment_gru_lightning.py --save_dir "gru_022" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 50 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_023" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 50 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_024" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 50 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_025" --epochs 50 --eval_interval 5 --log_interval 5 --lr 1e-3 --batch_size 32 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_026" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-2 --batch_size 1000 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_027" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 1000 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_028" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-3 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_029" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_030" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "wu_auc" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_031" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "wu_auc" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_032" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_033" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "contrastive" --loss_margin 0.1 --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_034" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_035" --epochs 1000 --pretrained_model "gru_029" --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_036" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 0.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_037" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_bce" --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_032" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 1
# python run_experiment_gru_lightning.py --save_dir "gru_038" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_bce" --wu_weight 0.9 --bce_weight 0.1 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_039" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "wu_auc" --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_040" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_asymmetric" --wu_weight 0.5 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_041" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_asymmetric_bce" --bce_weight 0.1 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_042" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_043" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "gru_044" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_bce" --wu_weight 0.9 --bce_weight 0.1 --num_workers 4

# With raw audio
# python run_experiment_gru_lightning.py --save_dir "raw_001" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "raw_002" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "raw_003" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-6 --weight_decay 1e-7 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "raw_004" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-8 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "raw_005" --pretrained_model "raw_002" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# Applying input normalization
# python run_experiment_gru_lightning.py --save_dir "raw_006" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# With Kaiming inizialization
# python run_experiment_gru_lightning.py --save_dir "raw_007" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# Adding new vansihing gradient improvement
# python run_experiment_gru_lightning.py --save_dir "raw_008" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# With sliding windows, LayerNorm, and residual connections
# python run_experiment_gru_lightning.py --save_dir "raw_009" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# With unified architecture for both raw and wav2vec modes
# python run_experiment_gru_lightning.py --save_dir "raw_010" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "wav2vec_001" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "wav2vec"
# python run_experiment_gru_lightning.py --save_dir "raw_009" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512

# python run_experiment_gru_lightning.py --save_dir "gru_045" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "wav2vec" --window_size 1024 --hop_size 512

# python run_experiment_gru_lightning.py --save_dir "raw_010" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-7 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "raw_011" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "wav2vec_001" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "wav2vec"
# python run_experiment_gru_lightning.py --save_dir "raw_011" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "raw_012" --epochs 1000 --pretrained_model "raw_011" --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512

# Improved command to prevent early convergence and improve AUC:
# python run_experiment_gru_lightning.py --save_dir "raw_013" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 5.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "raw_014" --epochs 10000 --eval_interval 100 --log_interval 100 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "raw" --window_size 1024 --hop_size 512

# python run_experiment_gru_lightning.py --save_dir "wav2vec_002" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "wav2vec" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "wav2vec_003" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "wu_auc" --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "wav2vec_004" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "wav2vec_005" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_asymmetric" --wu_weight 0.5 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "wav2vec_006" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_asymmetric_bce" --bce_weight 0.1 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "wav2vec_007" --epochs 200 --eval_interval 10 --log_interval 10 --lr 1e-4 --weight_decay 1e-5 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_bce" --wu_weight 0.9 --bce_weight 0.1 --num_workers 4

# python run_experiment_gru_lightning.py --save_dir "wav2vec_008" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "bce" --num_workers 4 --feature_mode "wav2vec" --window_size 1024 --hop_size 512
# python run_experiment_gru_lightning.py --save_dir "wav2vec_009" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "wu_auc" --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "wav2vec_010" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "asymmetric" --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "wav2vec_011" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_asymmetric" --wu_weight 0.5 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "wav2vec_012" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_asymmetric_bce" --bce_weight 0.1 --gamma_pos 1.0 --gamma_neg 4.0 --num_workers 4
# python run_experiment_gru_lightning.py --save_dir "wav2vec_013" --epochs 1000 --eval_interval 10 --log_interval 10 --lr 1e-5 --weight_decay 1e-6 --gradient_clip_val 10.0 --batch_size 100 --use_gpu --test_size 0.1 --dropout 0.1 --loss_fn "combined_wu_bce" --wu_weight 0.9 --bce_weight 0.1 --num_workers 4


import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torchmetrics.classification import MultilabelF1Score, MultilabelAveragePrecision, MultilabelAUROC
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa
from tqdm import tqdm
from pytorch_lightning.loggers import CSVLogger
import json
from utils import preprocess_audio, extract_wav2vec_embeddings, SAMPLE_RATE, TARGET_LENGTH, asymmetric_loss, MeanContrastiveRankingLoss, wu_auc_loss, combined_wu_bce_loss, combined_wu_asymmetric_loss, combined_asymmetric_bce_loss
import multiprocessing
import time
import shutil



class EmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, clip_ids, labels, indices=None, is_train=True, test_size=0.1, random_state=42):
        self.embedding_dir = embedding_dir
        self.is_train = is_train
        
        print(f"🔹 Looking for embedding files in: {os.path.abspath(embedding_dir)}")
        
        # Filter clip_ids to only include those with embedding files
        valid_indices = []
        valid_clip_ids = []
        valid_labels = []
        
        for idx, (clip_id, label) in enumerate(zip(clip_ids, labels)):
            embedding_path = os.path.join(embedding_dir, f"{clip_id}.npy")
            if os.path.exists(embedding_path):
                valid_indices.append(idx)
                valid_clip_ids.append(clip_id)
                valid_labels.append(label)
        
        if len(valid_clip_ids) == 0:
            raise ValueError(f"No embedding files found in {os.path.abspath(embedding_dir)}. "
                           f"Please check if the embedding files exist and are named correctly "
                           f"(should be named like 'clip_id.npy').")
        
        self.clip_ids = np.array(valid_clip_ids)
        self.labels = np.array(valid_labels)
        
        # Use provided indices if available, otherwise create train/test split
        if indices is not None:
            self.indices = indices
        else:
            # Create train/test split indices
            indices = np.arange(len(self.clip_ids))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            split_idx = int(len(indices) * (1 - test_size))
            
            if is_train:
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]
        
        print(f"🔹 {'Training' if is_train else 'Validation'} dataset size: {len(self.indices)}")
        
        # Additional safety checks
        if len(self.indices) == 0:
            raise ValueError(f"Empty {'training' if is_train else 'validation'} dataset. "
                           f"This might be due to an incorrect test_size parameter or insufficient data.")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        clip_idx = self.indices[idx]
        clip_id = self.clip_ids[clip_idx]
        label = self.labels[clip_idx]
        
        # Load embedding from file
        embedding_path = os.path.join(self.embedding_dir, f"{clip_id}.npy")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        embedding = np.load(embedding_path)
        
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class RawAudioDataset(Dataset):
    """Dataset class for raw audio files with learnable feature extraction."""
    
    def __init__(self, audio_dir, clip_ids, labels, indices=None, is_train=True, test_size=0.1, random_state=42, 
                 target_length=160000, sample_rate=16000, window_size=1024, hop_size=512):
        self.audio_dir = audio_dir
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.window_size = window_size  # Size of each time window
        self.hop_size = hop_size        # Stride between windows
        self.is_train = is_train
        
        print(f"🔹 Looking for raw audio files in: {os.path.abspath(audio_dir)}")
        print(f"🔹 Window size: {window_size}, Hop size: {hop_size}")
        
        # Store the pre-filtered data (no need to filter again)
        self.clip_ids = np.array(clip_ids)
        self.labels = np.array(labels)
        
        # Use provided indices if available, otherwise create train/test split
        if indices is not None:
            self.indices = indices
        else:
            # Create train/test split indices
            indices = np.arange(len(self.clip_ids))
            np.random.seed(random_state)
            np.random.shuffle(indices)
            split_idx = int(len(indices) * (1 - test_size))
            
            if is_train:
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]
        
        print(f"🔹 {'Training' if is_train else 'Validation'} dataset size: {len(self.indices)}")
        
        # Additional safety checks
        if len(self.indices) == 0:
            raise ValueError(f"Empty {'training' if is_train else 'validation'} dataset. "
                           f"This might be due to an incorrect test_size parameter or insufficient data.")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        clip_idx = self.indices[idx]
        clip_id = self.clip_ids[clip_idx]
        label = self.labels[clip_idx]
        
        # Find audio file (we know it exists since we pre-filtered)
        audio_path = None
        for ext in ['wav', 'mp3', 'flac', 'ogg']:
            temp_path = os.path.join(self.audio_dir, f"{clip_id}.{ext}")
            if os.path.exists(temp_path):
                audio_path = temp_path
                break
        
        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for {clip_id} - this should not happen with pre-filtered data")
        
        # Load and preprocess audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or truncate to target length
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.target_length]
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            return audio_tensor, torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {str(e)}")
            # Return zero tensor as fallback
            return torch.zeros(self.target_length, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class TrainEvalMetricsCallback(Callback):
    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def on_validation_epoch_end(self, trainer, pl_module):
        # Compute train metrics on training set
        pl_module.eval()
        device = pl_module.device
        loss_fn = pl_module.loss_fn

        # Train set metrics
        train_total_loss = 0.0
        train_total_samples = 0
        train_all_preds = []
        train_all_targets = []

        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                preds = pl_module(x)
                loss = loss_fn(preds, y)
                train_total_loss += loss.item() * x.size(0)
                train_total_samples += x.size(0)
                
                # Store predictions and targets
                train_all_preds.append(preds.detach())
                train_all_targets.append(y.detach())

        if train_total_samples == 0:
            print("⚠️ Warning: No training samples found for metrics computation")
            return

        train_avg_loss = train_total_loss / train_total_samples

        # Compute train metrics
        try:
            # Concatenate all predictions and targets
            train_all_preds = torch.cat(train_all_preds, dim=0)
            train_all_targets = torch.cat(train_all_targets, dim=0)
            
            # Apply sigmoid to raw logits for metrics computation
            train_all_preds_probs = torch.sigmoid(train_all_preds)
            train_all_preds_probs = torch.clamp(train_all_preds_probs, min=1e-7, max=1.0-1e-7)
            train_all_targets = train_all_targets.int()
            
            # Create temporary metrics for training data
            f1 = MultilabelF1Score(num_labels=pl_module.f1.num_labels, average="macro").to(device)
            map_metric = MultilabelAveragePrecision(num_labels=pl_module.map.num_labels, average="macro").to(device)
            auc = MultilabelAUROC(num_labels=pl_module.auc.num_labels, average="macro").to(device)
            
            # Compute train metrics using probabilities
            train_f1 = f1(train_all_preds_probs, train_all_targets)
            train_map = map_metric(train_all_preds_probs, train_all_targets)
            train_auc = auc(train_all_preds_probs, train_all_targets)
            
            print(f"✅ Epoch {trainer.current_epoch}: train_f1={train_f1:.4f}, train_map={train_map:.4f}, train_auc={train_auc:.4f}")
            
        except Exception as e:
            print(f"⚠️ Warning: Error computing train metrics: {str(e)}")
            train_f1 = torch.tensor(0.0)
            train_map = torch.tensor(0.0)
            train_auc = torch.tensor(0.0)

        # Compute all validation metrics on validation set
        val_total_loss = 0.0
        val_total_samples = 0
        val_all_preds = []
        val_all_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(device), y.to(device)
                preds = pl_module(x)
                loss = loss_fn(preds, y)
                val_total_loss += loss.item() * x.size(0)
                val_total_samples += x.size(0)
                
                # Store predictions and targets
                val_all_preds.append(preds.detach())
                val_all_targets.append(y.detach())

        if val_total_samples == 0:
            print("⚠️ Warning: No validation samples found for validation metrics computation")
            return

        val_avg_loss = val_total_loss / val_total_samples

        # Compute all validation metrics
        try:
            # Concatenate all predictions and targets
            val_all_preds = torch.cat(val_all_preds, dim=0)
            val_all_targets = torch.cat(val_all_targets, dim=0)
            
            # Apply sigmoid to raw logits for metrics computation
            val_all_preds_probs = torch.sigmoid(val_all_preds)
            val_all_preds_probs = torch.clamp(val_all_preds_probs, min=1e-7, max=1.0-1e-7)
            val_all_targets = val_all_targets.int()
            
            # Compute all validation metrics using probabilities
            val_f1 = f1(val_all_preds_probs, val_all_targets)
            val_map = map_metric(val_all_preds_probs, val_all_targets)
            val_auc = auc(val_all_preds_probs, val_all_targets)
            
            print(f"✅ Epoch {trainer.current_epoch}: val_f1={val_f1:.4f}, val_map={val_map:.4f}, val_auc={val_auc:.4f}")
            
        except Exception as e:
            print(f"⚠️ Warning: Error computing validation metrics: {str(e)}")
            val_f1 = torch.tensor(0.0)
            val_map = torch.tensor(0.0)
            val_auc = torch.tensor(0.0)

        # Log all metrics using the trainer's logger
        trainer.logger.log_metrics({
            "epoch": trainer.current_epoch,
            "train_loss_eval": train_avg_loss,
            "train_f1_eval": train_f1.item(),
            "train_map_eval": train_map.item(),
            "train_auc_eval": train_auc.item(),
            "val_loss_eval": val_avg_loss,
            "val_f1_eval": val_f1.item(),
            "val_map_eval": val_map.item(),
            "val_auc_eval": val_auc.item()
        }, step=trainer.current_epoch)

        pl_module.train()  # Switch back to training mode

class WeightNormCallback(Callback):
    def __init__(self):
        super().__init__()
        self.grad_norms = []
        self.layer_grad_norms = {}
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Capture gradient norms after each training batch (before optimizer step)
        if batch_idx % trainer.log_every_n_steps == 0:  # Only capture periodically to avoid overhead
            total_grad_norm = 0.0
            grad_count = 0
            
            # Track gradients per layer to identify where vanishing occurs
            layer_grads = {}
            
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_count += 1
                    grad_norm = torch.norm(param.grad.data, p=2).item()
                    total_grad_norm += grad_norm ** 2
                    
                    # Track gradients by layer
                    layer_name = name.split('.')[0]  # Get the main layer name
                    if layer_name not in layer_grads:
                        layer_grads[layer_name] = []
                    layer_grads[layer_name].append(grad_norm)
            
            if grad_count > 0:
                total_grad_norm = total_grad_norm ** 0.5
                self.grad_norms.append(total_grad_norm)
                
                # Store layer gradients
                for layer_name, grads in layer_grads.items():
                    if layer_name not in self.layer_grad_norms:
                        self.layer_grad_norms[layer_name] = []
                    self.layer_grad_norms[layer_name].append(np.mean(grads))
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate overall L2 norm for all weights
        total_weight_norm = 0.0
        param_count = 0
        
        for name, param in pl_module.named_parameters():
            if param.requires_grad:  # Only compute for trainable parameters
                param_count += 1
                # Accumulate weight norm
                total_weight_norm += torch.norm(param.data, p=2).item() ** 2
        
        # Take square root to get the overall L2 norm
        total_weight_norm = total_weight_norm ** 0.5
        
        # Use the average gradient norm from this epoch
        total_grad_norm = np.mean(self.grad_norms) if self.grad_norms else 0.0
        
        # Log the overall norms
        trainer.logger.log_metrics({
            "epoch": trainer.current_epoch,
            "total_weight_norm": total_weight_norm,
            "total_grad_norm": total_grad_norm
        }, step=trainer.current_epoch)
        
        # Add monitoring every 10 epochs
        if trainer.current_epoch % 10 == 0:
            print(f"🔍 Epoch {trainer.current_epoch}: Total weight norm = {total_weight_norm:.6f}, Total gradient norm = {total_grad_norm:.6f}")
            
            # Check for gradient explosion/vanishing
            if total_grad_norm > 10.0:
                print(f"⚠️ Warning: High gradient norm detected: {total_grad_norm:.6f}")
            elif total_grad_norm < 1e-6:
                print(f"⚠️ Warning: Very low gradient norm detected: {total_grad_norm:.6f}")
                
                # Print layer-specific gradient norms to identify the problem
                print("🔍 Layer gradient norms:")
                for layer_name, grads in self.layer_grad_norms.items():
                    avg_grad = np.mean(grads) if grads else 0.0
                    print(f"   {layer_name}: {avg_grad:.6f}")
        
        # Clear the gradient norms for the next epoch
        self.grad_norms.clear()
        self.layer_grad_norms.clear()

# ========================
# 1. Parse Input Arguments
# ========================
parser = argparse.ArgumentParser(description="Train an audio classification model with Wav2Vec2 embeddings and RNN (Lightning).")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--eval_interval", type=int, default=100, help="Interval for evaluating the model")
parser.add_argument("--log_interval", type=int, default=100, help="Interval for logging metrics")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
parser.add_argument("--test_size", type=float, default=0.1, help="Test size")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
parser.add_argument("--save_dir", type=str, default="results", help="Directory to save the model and metrics")
parser.add_argument("--pretrained_model", type=str, default=None, help="Path to a pretrained model checkpoint")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
parser.add_argument("--embedding_dir", type=str, default="embeddings", help="Directory to load/save embeddings")
parser.add_argument("--audio_dir", type=str, default="../tmp/fsd50k/FSD50K.dev_audio", help="Directory containing raw audio files")
parser.add_argument("--feature_mode", type=str, default="wav2vec", choices=["wav2vec", "raw"], 
                   help="Feature extraction mode: 'wav2vec' for pre-computed embeddings, 'raw' for learnable feature extraction")
parser.add_argument("--window_size", type=int, default=1024, help="Window size for raw audio feature extraction")
parser.add_argument("--hop_size", type=int, default=512, help="Hop size (stride) between windows for raw audio feature extraction")
parser.add_argument("--loss_fn", type=str, default="bce", choices=["bce", "asymmetric", "contrastive", "wu_auc", "combined_wu_bce", "combined_wu_asymmetric", "combined_asymmetric_bce"], 
                   help="Loss function to use: bce, asymmetric, contrastive, wu_auc, combined_wu_bce, combined_wu_asymmetric, or combined_asymmetric_bce")
parser.add_argument("--loss_margin", type=float, default=0.1, help="Margin for contrastive loss or Wu AUC loss")
parser.add_argument("--gamma_pos", type=float, default=0.0, help="Gamma positive for asymmetric loss")
parser.add_argument("--gamma_neg", type=float, default=4.0, help="Gamma negative for asymmetric loss")
parser.add_argument("--wu_weight", type=float, default=0.5, help="Weight for Wu AUC component in combined loss")
parser.add_argument("--bce_weight", type=float, default=0.5, help="Weight for BCE component in combined loss")
parser.add_argument("--gradient_clip_val", type=float, default=0.5, help="Gradient clipping value to prevent gradient explosion/vanishing")
parser.add_argument("--use_scheduler", action="store_true", help="Use cosine annealing scheduler")
parser.add_argument("--use_early_stopping", action="store_true", help="Use early stopping callback")
parser.add_argument("--early_stopping_patience", type=int, default=100, help="Patience for early stopping")
args = parser.parse_args()

# ========================
# 2. Device
# ========================
device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

# Check GPU compatibility
if args.use_gpu and torch.cuda.is_available():
    try:
        # Test GPU compatibility by creating a small tensor
        test_tensor = torch.tensor([1.0], device=device)
        print(f"\n🔹 Using device: {device}")
        print(f"🔹 GPU: {torch.cuda.get_device_name()}")
        print(f"🔹 CUDA version: {torch.version.cuda}")
        print(f"🔹 GPU capability: {torch.cuda.get_device_capability()}")
        
        # Check GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"🔹 GPU memory: {gpu_memory:.1f} GB")
        
    except Exception as e:
        print(f"⚠️ Warning: GPU compatibility issue detected: {str(e)}")
        print("🔹 Falling back to CPU")
        device = torch.device("cpu")
        args.use_gpu = False
else:
    print(f"\n🔹 Using device: {device}")

# Enable Tensor Cores for better performance
if args.use_gpu and torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    print("🔹 Enabled Tensor Cores for better performance")

# ========================
# 3. Load Wav2Vec 2.0 Model
# ========================
MODEL_NAME = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
wav2vec_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
wav2vec_model.eval()
wav2vec_model.to(device)

TARGET_LENGTH = 10 * 16000
SAMPLE_RATE = 16000

# Ensure save_dir is a directory
if os.path.exists(args.save_dir):
    if not os.path.isdir(args.save_dir):
        print(f"⚠️ {args.save_dir} exists as a file. Removing it to create a directory.")
        os.remove(args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
else:
    os.makedirs(args.save_dir, exist_ok=True)

# Save args to JSON for reproducibility
with open(os.path.join(args.save_dir, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

# ========================
# 5. Load Dataset & Extract Features
# ========================
csv_path = "../tmp/fsd50k_spc/fsd50k_clips_labels_duration_max10sec.csv"
print(f"🔹 Loading CSV from: {csv_path}")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path)
clip_ids = df["clip_id"].values
labels = df.iloc[:, 2:-1].values
AUDIO_DIR = args.audio_dir
print(f"🔹 Audio directory: {AUDIO_DIR}")
if not os.path.exists(AUDIO_DIR):
    raise FileNotFoundError(f"Audio directory not found at: {AUDIO_DIR}")

print(f"🔹 Number of clips in CSV: {len(clip_ids)}")
print(f"🔹 Feature extraction mode: {args.feature_mode}")

if args.feature_mode == "wav2vec":
    # Wav2Vec mode: Load or extract embeddings
    embedding_dir = args.embedding_dir
    # Use the embeddings directory directly
    embeddings_subdir = embedding_dir
    os.makedirs(embeddings_subdir, exist_ok=True)

    print(f"🔹 Checking for precomputed embeddings in: {embeddings_subdir}")

    # Check if embedding files exist directly
    embedding_files = [f for f in os.listdir(embeddings_subdir) if f.endswith('.npy')]
    if len(embedding_files) > 0:
        print(f"🔹 Found {len(embedding_files)} precomputed embedding files")
        
        # Try to load metadata if it exists
        if os.path.exists(os.path.join(embeddings_subdir, "metadata.json")):
            print("🔹 Loading precomputed embeddings metadata...")
            with open(os.path.join(embeddings_subdir, "metadata.json"), "r") as f:
                metadata = json.load(f)
            print(f"🔹 Metadata indicates {metadata.get('total_samples', 'unknown')} embeddings")
        else:
            print("🔹 No metadata.json found, but embedding files exist")
            # Create basic metadata from existing files
            metadata = {
                "total_samples": len(embedding_files),
                "embedding_files": embedding_files[:10]  # Store first 10 as example
            }
            print(f"🔹 Using {len(embedding_files)} existing embedding files")
    else:
        print("🔹 No precomputed embeddings found. Starting extraction...")
        processed_count = 0
        error_count = 0
        missing_files = []
        
        # Create embedding directory
        os.makedirs(embeddings_subdir, exist_ok=True)
        
        for clip_id, label in tqdm(zip(clip_ids, labels), total=len(clip_ids)):
            audio_path = os.path.join(AUDIO_DIR, f"{clip_id}.wav")
            if os.path.exists(audio_path):
                try:
                    emb = extract_wav2vec_embeddings(audio_path, processor, wav2vec_model, device)
                    # Save individual embedding in the subdirectory
                    embedding_path = os.path.join(embeddings_subdir, f"{clip_id}.npy")
                    np.save(embedding_path, emb)
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"🔹 Processed {processed_count} files")
                except Exception as e:
                    print(f"Warning: Error processing {clip_id}: {str(e)}")
                    error_count += 1
            else:
                missing_files.append(clip_id)
                error_count += 1
        
        print(f"🔹 Processed {processed_count} files successfully")
        print(f"🔹 Encountered {error_count} errors")
        print(f"🔹 Missing files: {len(missing_files)}")
        
        if processed_count == 0:
            raise ValueError("No audio files were successfully processed. Please check the audio directory path and file permissions.")
        
        # Save metadata
        metadata = {
            "total_samples": processed_count,
            "embedding_shape": emb.shape,
            "label_shape": labels.shape[1:]
        }
        with open(os.path.join(embeddings_subdir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("🔹 Saved embeddings and metadata for future runs.")

# Create datasets based on feature mode
try:
    print(f"🔹 Creating datasets with test_size={args.test_size}")
    print(f"🔹 Total clip_ids: {len(clip_ids)}")
    print(f"🔹 Total labels shape: {labels.shape}")
    
    if args.feature_mode == "wav2vec":
        # Check if embedding files exist
        embedding_files = [f for f in os.listdir(embeddings_subdir) if f.endswith('.npy')]
        print(f"🔹 Found {len(embedding_files)} embedding files in {embeddings_subdir}")
        
        if len(embedding_files) == 0:
            raise ValueError(f"No embedding files found in {embeddings_subdir}. "
                            f"Please run the embedding extraction first.")
        
        # First, filter clip_ids to only include those with embedding files
        valid_indices = []
        valid_clip_ids = []
        valid_labels = []
        
        for idx, (clip_id, label) in enumerate(zip(clip_ids, labels)):
            embedding_path = os.path.join(embeddings_subdir, f"{clip_id}.npy")
            if os.path.exists(embedding_path):
                valid_indices.append(idx)
                valid_clip_ids.append(clip_id)
                valid_labels.append(label)
        
        if len(valid_clip_ids) == 0:
            raise ValueError(f"No valid embedding files found. Please check the embedding directory.")
        
        print(f"🔹 Valid clip_ids: {len(valid_clip_ids)}")
        
        # Create train/test split on the filtered data
        indices = np.arange(len(valid_clip_ids))
        np.random.seed(42)  # Fixed seed for reproducibility
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - args.test_size))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        print(f"🔹 Train split size: {len(train_indices)}")
        print(f"🔹 Validation split size: {len(val_indices)}")
        
        if len(train_indices) == 0:
            raise ValueError("Training split is empty. This might be due to an incorrect test_size parameter.")
        if len(val_indices) == 0:
            raise ValueError("Validation split is empty. This might be due to an incorrect test_size parameter.")
        
        # Create datasets with pre-filtered data
        train_dataset = EmbeddingDataset(embeddings_subdir, valid_clip_ids, valid_labels, indices=train_indices, is_train=True, test_size=0.0)
        val_dataset = EmbeddingDataset(embeddings_subdir, valid_clip_ids, valid_labels, indices=val_indices, is_train=False, test_size=0.0)
        
    else:  # raw audio mode
        print(f"🔹 Using raw audio files from {AUDIO_DIR}")
        
        # FIRST: Filter for available audio files before creating train/test split
        print(f"🔹 Filtering for available audio files...")
        valid_indices = []
        valid_clip_ids = []
        valid_labels = []
        
        for idx, (clip_id, label) in enumerate(zip(clip_ids, labels)):
            # Try different audio formats
            audio_found = False
            for ext in ['wav', 'mp3', 'flac', 'ogg']:
                audio_path = os.path.join(AUDIO_DIR, f"{clip_id}.{ext}")
                if os.path.exists(audio_path):
                    valid_indices.append(idx)
                    valid_clip_ids.append(clip_id)
                    valid_labels.append(label)
                    audio_found = True
                    break
        
        print(f"🔹 Found {len(valid_clip_ids)} audio files out of {len(clip_ids)} total clips")
        
        if len(valid_clip_ids) == 0:
            raise ValueError(f"No audio files found in {AUDIO_DIR}")
        
        # SECOND: Create train/test split on the filtered data
        indices = np.arange(len(valid_clip_ids))
        np.random.seed(42)  # Fixed seed for reproducibility
        np.random.shuffle(indices)
        split_idx = int(len(indices) * (1 - args.test_size))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        print(f"🔹 Train split size: {len(train_indices)}")
        print(f"🔹 Validation split size: {len(val_indices)}")
        
        if len(train_indices) == 0:
            raise ValueError("Training split is empty. This might be due to an incorrect test_size parameter.")
        if len(val_indices) == 0:
            raise ValueError("Validation split is empty. This might be due to an incorrect test_size parameter.")
        
        # Create datasets with raw audio using the filtered data
        train_dataset = RawAudioDataset(AUDIO_DIR, valid_clip_ids, valid_labels, indices=train_indices, is_train=True, test_size=0.0, window_size=args.window_size, hop_size=args.hop_size)
        val_dataset = RawAudioDataset(AUDIO_DIR, valid_clip_ids, valid_labels, indices=val_indices, is_train=False, test_size=0.0, window_size=args.window_size, hop_size=args.hop_size)
    
    print(f"🔹 Train dataset size: {len(train_dataset)}")
    print(f"🔹 Validation dataset size: {len(val_dataset)}")
    
    # Additional safety checks
    if len(train_dataset) < 10:
        raise ValueError(f"Training dataset too small ({len(train_dataset)} samples). Need at least 10 samples.")
    if len(val_dataset) < 5:
        raise ValueError(f"Validation dataset too small ({len(val_dataset)} samples). Need at least 5 samples.")
    
    print(f"✅ Dataset creation successful!")
    
except Exception as e:
    print(f"❌ Error creating datasets: {str(e)}")
    print("\nPossible solutions:")
    if args.feature_mode == "wav2vec":
        print("1. Check if the embedding files exist in the correct directory")
        print("2. Make sure the embedding files are named correctly (clip_id.npy)")
    else:
        print("1. Check if the audio files exist in the correct directory")
        print("2. Make sure the audio files are named correctly (clip_id.wav/mp3/flac/ogg)")
    print("3. Verify that the test_size parameter is appropriate")
    print("4. Check if the CSV file contains the correct clip_ids")
    raise

# Create dataloaders
# Use user-provided parameters without modification
adjusted_batch_size = args.batch_size
adjusted_num_workers = args.num_workers

train_loader = DataLoader(
    train_dataset, 
    batch_size=adjusted_batch_size, 
    shuffle=True,
    num_workers=adjusted_num_workers,
    pin_memory=True,
    persistent_workers=True if adjusted_num_workers > 0 else False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=adjusted_batch_size,
    num_workers=adjusted_num_workers,
    pin_memory=True,
    persistent_workers=True if adjusted_num_workers > 0 else False
)

# ========================
# 7. Lightning Model
# ========================
class LitRNNClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, lr, weight_decay, dropout, 
                 loss_fn="bce", loss_margin=0.1, gamma_pos=0.0, gamma_neg=4.0, wu_weight=0.5, bce_weight=0.5,
                 feature_mode="wav2vec", window_size=1024, hop_size=512, use_scheduler=False):
        super().__init__()
        self.save_hyperparameters()
        self.feature_mode = feature_mode
        self.window_size = window_size
        self.hop_size = hop_size
        
        # Feature extraction layers (for both modes to maintain consistency)
        if feature_mode == "raw":
            # Learnable feature extraction: raw audio window -> 768 features
            # Use reduced dropout in feature extractor to prevent vanishing gradients
            feature_dropout = max(0.05, dropout * 0.3)  # Further reduce dropout for better gradient flow
            
            # Create simplified feature extractor to prevent gradient issues
            self.feature_extractor = nn.ModuleList([
                nn.Linear(window_size, 768),  # Direct mapping to avoid intermediate layers
            ])
            
            # Normalization and activation layers
            self.feature_norms = nn.ModuleList([
                nn.LayerNorm(768),
            ])
            
            self.feature_activations = nn.ModuleList([
                nn.LeakyReLU(0.1),
            ])
            
            self.feature_dropouts = nn.ModuleList([
                nn.Dropout(feature_dropout),
            ])
            # Update input dimension for GRU
            gru_input_dim = 768
        else:
            # Wav2Vec mode: add a lightweight feature projection layer for consistency
            # This ensures both architectures have similar structure and training dynamics
            feature_dropout = max(0.05, dropout * 0.3)  # Same dropout strategy
            
            # Lightweight projection: 768 -> 768 (identity-like transformation)
            # This maintains the same architecture pattern without changing the data
            self.feature_extractor = nn.ModuleList([
                nn.Linear(input_dim, input_dim)  # Identity-like projection
            ])
            
            # Normalization and activation layers (same as raw mode)
            self.feature_norms = nn.ModuleList([
                nn.LayerNorm(input_dim)
            ])
            
            self.feature_activations = nn.ModuleList([
                nn.LeakyReLU(0.1)
            ])
            
            self.feature_dropouts = nn.ModuleList([
                nn.Dropout(feature_dropout)
            ])
            gru_input_dim = input_dim
        
        # Only apply dropout if num_layers > 1
        gru_dropout = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers, batch_first=True, dropout=gru_dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),  # Use LayerNorm instead of BatchNorm for better gradient flow
            nn.LeakyReLU(0.1),  # Use LeakyReLU instead of ReLU
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
            # Removed Sigmoid - we'll work with raw logits for better gradient flow
        )
        
        # Initialize weights properly to avoid zero predictions
        self._init_weights()
        
        # Initialize loss function based on the argument
        if loss_fn == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for raw logits
        elif loss_fn == "asymmetric":
            # Use a simpler asymmetric loss configuration
            self.loss_fn = lambda preds, targets: asymmetric_loss(
                preds, targets, gamma_pos=0.0, gamma_neg=2.0, margin=0.05
            )
        elif loss_fn == "contrastive":
            self.loss_fn = MeanContrastiveRankingLoss(margin=loss_margin)
        elif loss_fn == "wu_auc":
            self.loss_fn = lambda preds, targets: wu_auc_loss(preds, targets, margin=loss_margin)
        elif loss_fn == "combined_wu_bce":
            self.loss_fn = lambda preds, targets: combined_wu_bce_loss(
                preds, targets, wu_weight=wu_weight, bce_weight=bce_weight, margin=loss_margin
            )
        elif loss_fn == "combined_wu_asymmetric":
            self.loss_fn = lambda preds, targets: combined_wu_asymmetric_loss(
                preds, targets, wu_weight=wu_weight, asymmetric_weight=1-wu_weight, 
                margin=loss_margin, gamma_pos=gamma_pos, gamma_neg=gamma_neg, asymmetric_margin=0.05
            )
        elif loss_fn == "combined_asymmetric_bce":
            self.loss_fn = lambda preds, targets: combined_asymmetric_bce_loss(
                preds, targets, asymmetric_weight=1-bce_weight, bce_weight=bce_weight,
                gamma_pos=gamma_pos, gamma_neg=gamma_neg, margin=0.05
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
            
        self.f1 = MultilabelF1Score(num_labels=num_classes, average="macro")
        self.map = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.auc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.training_step_outputs = []

    def forward(self, x):
        # Apply feature extraction for both modes to maintain consistency
        if self.feature_mode == "raw":
            # x shape: (batch_size, audio_length)
            batch_size, audio_length = x.shape
            
            # NORMALIZE INPUT: This is crucial for preventing vanishing gradients
            # Normalize to zero mean and unit variance per batch
            x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
            
            # Use unfold to create sliding windows more efficiently
            # This maintains better gradient flow than the loop approach
            x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, audio_length)
            
            # Create sliding windows using unfold
            # This is equivalent to the loop but much more efficient and gradient-friendly
            x = x.unfold(dimension=2, size=self.window_size, step=self.hop_size)
            # Result: (batch_size, 1, num_windows, window_size)
            
            # Reshape for batch processing: (batch_size * num_windows, window_size)
            num_windows = x.size(2)
            x = x.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_windows, 1, window_size)
            x = x.view(batch_size * num_windows, self.window_size)
            
            # Extract features for all windows at once (simplified)
            x = self.feature_extractor[0](x)  # (batch_size * num_windows, 768)
            x = self.feature_norms[0](x)
            x = self.feature_activations[0](x)
            x = self.feature_dropouts[0](x)
            
            # Reshape back to sequence: (batch_size, num_windows, 768)
            x = x.view(batch_size, num_windows, -1)
            
            # Monitor feature extractor output for vanishing gradients (only occasionally)
            if hasattr(self, 'training_step_outputs') and len(self.training_step_outputs) % 100 == 0:
                if torch.all(x == 0) or torch.all(torch.abs(x) < 1e-6):
                    print(f"⚠️ Warning: Feature extractor output is near zero! Max abs value: {torch.abs(x).max():.6f}")
                print(f"🔍 Raw audio: {audio_length} samples -> {num_windows} windows -> {x.shape}")
            
        else:
            # Wav2Vec mode: x is already (batch_size, seq_len, 768)
            # Apply the same feature processing pattern for consistency
            batch_size, seq_len, features = x.shape
            
            # Reshape for batch processing: (batch_size * seq_len, features)
            x = x.view(batch_size * seq_len, features)
            
            # Apply the same feature processing pattern (lightweight transformation)
            x = self.feature_extractor[0](x)  # (batch_size * seq_len, features)
            x = self.feature_norms[0](x)
            x = self.feature_activations[0](x)
            x = self.feature_dropouts[0](x)
            
            # Reshape back to sequence: (batch_size, seq_len, features)
            x = x.view(batch_size, seq_len, -1)
            
            # Monitor feature extractor output for vanishing gradients (only occasionally)
            if hasattr(self, 'training_step_outputs') and len(self.training_step_outputs) % 100 == 0:
                if torch.all(x == 0) or torch.all(torch.abs(x) < 1e-6):
                    print(f"⚠️ Warning: Feature extractor output is near zero! Max abs value: {torch.abs(x).max():.6f}")
                print(f"🔍 Wav2Vec: {seq_len} frames -> {x.shape}")
        
        # Monitor input to GRU (only occasionally)
        if hasattr(self, 'training_step_outputs') and len(self.training_step_outputs) % 100 == 0:
            if torch.all(x == 0) or torch.all(torch.abs(x) < 1e-6):
                print(f"⚠️ Warning: GRU input is near zero! Max abs value: {torch.abs(x).max():.6f}")
        
        _, h_n = self.gru(x)
        h_n = h_n[-1]
        
        # Monitor GRU output (only occasionally)
        if hasattr(self, 'training_step_outputs') and len(self.training_step_outputs) % 100 == 0:
            if torch.all(h_n == 0) or torch.all(torch.abs(h_n) < 1e-6):
                print(f"⚠️ Warning: GRU output is near zero! Max abs value: {torch.abs(h_n).max():.6f}")
        
        output = self.fc(h_n)
        
        # Add safety check for output (only occasionally)
        if hasattr(self, 'training_step_outputs') and len(self.training_step_outputs) % 100 == 0:
            if torch.all(output == 0):
                print(f"⚠️ Warning: All predictions are zero! Input shape: {x.shape}, Output shape: {output.shape}")
                print(f"   GRU output range: [{h_n.min():.4f}, {h_n.max():.4f}]")
                print(f"   FC output range: [{self.fc(h_n).min():.4f}, {self.fc(h_n).max():.4f}]")
            print(f"🔍 Model output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
        
        return output

    def _init_weights(self):
        """Initialize weights to prevent vanishing gradients"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use Kaiming initialization for LeakyReLU activations
                if 'feature_extractor' in name:
                    # Special initialization for feature extractor layers (both modes)
                    if self.feature_mode == "raw":
                        # Raw mode: standard Kaiming initialization
                        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                    else:
                        # Wav2Vec mode: identity-like initialization for the projection layer
                        nn.init.eye_(module.weight)  # Identity matrix initialization
                        # Don't override with zeros - keep the identity matrix
                    
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.01)  # Small positive bias to avoid dead neurons
                else:
                    # Standard initialization for other linear layers
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # Initialize layer norm layers properly
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.GRU):
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        # Input-to-hidden weights - use Xavier for better gradient flow
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in param_name:
                        # Hidden-to-hidden weights - use orthogonal initialization for RNNs
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in param_name:
                        # Initialize bias with small positive values to avoid dead neurons
                        nn.init.constant_(param, 0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Validate input data
        if batch_idx == 0:  # Only check first batch to avoid spam
            print(f"🔍 Training batch {batch_idx}: x shape={x.shape}, y shape={y.shape}")
            print(f"   x range: [{x.min():.4f}, {x.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
            print(f"   y sum: {y.sum().item()}, y total: {y.numel()}")
            
            # Check for data corruption
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                print("❌ ERROR: Input data contains NaN or Inf values!")
            if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                print("❌ ERROR: Target data contains NaN or Inf values!")
            if y.sum() > y.numel():
                print("❌ ERROR: Target sum exceeds total elements (data corruption)!")
        
        preds = self(x)
        
        # Add safety checks for loss computation
        try:
            loss = self.loss_fn(preds, y)
            # Check if loss is finite
            if not torch.isfinite(loss):
                print(f"⚠️ Warning: Non-finite loss detected: {loss.item()}")
                # Use a fallback loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        except Exception as e:
            print(f"⚠️ Warning: Error computing loss: {str(e)}")
            # Use a fallback loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        
        self.training_step_outputs.append(loss.item())
        
        # Log based on the log_interval parameter
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Log average training loss for the epoch
        avg_loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Validate input data
        if batch_idx == 0:  # Only check first batch to avoid spam
            print(f"🔍 Validation batch {batch_idx}: x shape={x.shape}, y shape={y.shape}")
            print(f"   x range: [{x.min():.4f}, {x.max():.4f}], y range: [{y.min():.4f}, {y.max():.4f}]")
            print(f"   y sum: {y.sum().item()}, y total: {y.numel()}")
            
            # Check for data corruption
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                print("❌ ERROR: Validation input data contains NaN or Inf values!")
            if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
                print("❌ ERROR: Validation target data contains NaN or Inf values!")
            if y.sum() > y.numel():
                print("❌ ERROR: Validation target sum exceeds total elements (data corruption)!")
        
        preds = self(x)
        
        # Add safety checks for loss computation
        try:
            loss = self.loss_fn(preds, y)
            # Check if loss is finite
            if not torch.isfinite(loss):
                print(f"⚠️ Warning: Non-finite loss detected: {loss.item()}")
                # Use a fallback loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        except Exception as e:
            print(f"⚠️ Warning: Error computing loss: {str(e)}")
            # Use a fallback loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store predictions and targets for epoch-end computation
        if not hasattr(self, 'val_preds'):
            self.val_preds = []
            self.val_targets = []
        
        self.val_preds.append(preds.detach())
        self.val_targets.append(y.detach())
        
        return loss

    def on_validation_epoch_end(self):
        # Compute and log final metrics for the epoch
        if hasattr(self, 'val_preds') and len(self.val_preds) > 0:
            try:
                # Concatenate all predictions and targets
                all_preds = torch.cat(self.val_preds, dim=0)
                all_targets = torch.cat(self.val_targets, dim=0)
                
                # Validate data before computing metrics
                if torch.any(torch.isnan(all_preds)) or torch.any(torch.isinf(all_preds)):
                    print("❌ ERROR: Validation predictions contain NaN or Inf values!")
                    return
                
                if torch.any(torch.isnan(all_targets)) or torch.any(torch.isinf(all_targets)):
                    print("❌ ERROR: Validation targets contain NaN or Inf values!")
                    return
                
                # Check for data corruption
                if all_targets.sum() > all_targets.numel():
                    print(f"❌ ERROR: Validation target sum ({all_targets.sum()}) exceeds total elements ({all_targets.numel()})!")
                    return
                
                # Apply sigmoid to raw logits for metrics computation
                all_preds_probs = torch.sigmoid(all_preds)
                all_preds_probs = torch.clamp(all_preds_probs, min=1e-7, max=1.0-1e-7)
                all_targets = all_targets.int()
                
                # Check if we have any positive labels
                if all_targets.sum() == 0:
                    print("⚠️ Warning: No positive labels in validation set - skipping metrics computation")
                    return
                
                # Compute metrics using probabilities
                val_f1 = self.f1(all_preds_probs, all_targets)
                val_map = self.map(all_preds_probs, all_targets)
                val_auc = self.auc(all_preds_probs, all_targets)
                
                # Log the metrics
                self.log('val_f1', val_f1, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_map', val_map, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val_auc', val_auc, on_step=False, on_epoch=True, prog_bar=True)
                
                print(f"✅ Epoch {self.current_epoch}: val_f1={val_f1:.4f}, val_map={val_map:.4f}, val_auc={val_auc:.4f}")
                
            except Exception as e:
                print(f"⚠️ Warning: Error computing validation metrics: {str(e)}")
                print(f"   Predictions shape: {all_preds.shape}, Targets shape: {all_targets.shape}")
                print(f"   Predictions range: [{all_preds.min():.4f}, {all_preds.max():.4f}]")
                print(f"   Targets sum: {all_targets.sum()}, Targets total: {all_targets.numel()}")
        
        # Clear stored predictions and targets
        if hasattr(self, 'val_preds'):
            self.val_preds.clear()
            self.val_targets.clear()

    def configure_optimizers(self):
        # Use lower learning rate for raw mode to prevent gradient issues
        if self.feature_mode == "raw":
            lr = self.hparams.lr * 0.1  # Reduce learning rate for raw audio
        else:
            lr = self.hparams.lr
        
        # Create optimizer with better parameters
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=lr, 
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Conditionally add cosine annealing scheduler
        if self.hparams.use_scheduler:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=50,  # Restart every 50 epochs
                T_mult=2,  # Double the restart interval each time
                eta_min=lr * 0.001  # Minimum LR
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer

# ========================
# 8. Training
# ========================
# Determine input dimension based on feature mode
if args.feature_mode == "wav2vec":
    # Wav2Vec mode: input is embeddings (768 features)
    input_dim = train_dataset[0][0].shape[1]
else:
    # Raw audio mode: input is raw audio samples
    input_dim = train_dataset[0][0].shape[0]  # audio length

print(f"🔹 Feature mode: {args.feature_mode}")
print(f"🔹 Input dimension: {input_dim}")
print(f"🔹 Number of classes: {train_dataset[0][1].shape[0]}")

model = LitRNNClassifier(
    input_dim=input_dim,
    hidden_dim=256,
    num_layers=2,  # Changed from 1 to 2 to properly utilize dropout
    num_classes=train_dataset[0][1].shape[0],
    lr=args.lr,
    weight_decay=args.weight_decay,
    dropout=args.dropout,
    loss_fn=args.loss_fn,
    loss_margin=args.loss_margin,
    gamma_pos=args.gamma_pos,
    gamma_neg=args.gamma_neg,
    wu_weight=args.wu_weight,
    bce_weight=args.bce_weight,
    feature_mode=args.feature_mode,
    window_size=args.window_size,
    hop_size=args.hop_size,
    use_scheduler=args.use_scheduler
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=args.save_dir,
    filename='best-checkpoint',
    save_top_k=1,
    mode='min'
)

# Conditionally add early stopping callback
callbacks = [checkpoint_callback]
if args.use_early_stopping:
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)

# Configure CSV logger with reduced logging frequency
csv_logger = CSVLogger(
    save_dir=args.save_dir,
    name="metrics",
    version=None,  # Don't create new version directories
    flush_logs_every_n_steps=args.log_interval  # Use log_interval parameter
)
train_eval_callback = TrainEvalMetricsCallback(train_loader, val_loader)  # Pass both loaders for train metrics on train and val sets
weight_norm_callback = WeightNormCallback()  # Add weight norm callback

# Add metrics callbacks to the list
callbacks.extend([weight_norm_callback, train_eval_callback])

# Disable sanity check if validation dataset is too small
num_sanity_val_steps = 0 if len(val_dataset) < args.batch_size else 2

# Configure trainer based on device
if args.use_gpu and torch.cuda.is_available():
    try:
        # Test GPU compatibility first
        test_tensor = torch.randn(10, 10).cuda()
        test_gru = torch.nn.GRU(10, 5, 1, batch_first=True).cuda()
        test_input = torch.randn(2, 3, 10).cuda()
        test_output, _ = test_gru(test_input)
        print("✅ GPU compatibility test passed")
        trainer_config = {
            'accelerator': 'gpu',
            'devices': 1
        }
    except Exception as e:
        print(f"⚠️ GPU compatibility test failed: {e}")
        print("🔹 Falling back to CPU training")
        trainer_config = {
            'accelerator': 'cpu',
            'devices': 1
        }
        device = torch.device("cpu")
        args.use_gpu = False
else:
    trainer_config = {
        'accelerator': 'cpu',
        'devices': 1  # Use 1 CPU device instead of None
    }

trainer = pl.Trainer(
    max_epochs=args.epochs,
    callbacks=callbacks,
    default_root_dir=args.save_dir,
    logger=csv_logger,
    check_val_every_n_epoch=args.eval_interval,
    log_every_n_steps=args.log_interval,
    gradient_clip_val=args.gradient_clip_val,  # Use command line parameter for gradient clipping
    num_sanity_val_steps=num_sanity_val_steps,  # Disable sanity check for small validation sets
    **trainer_config
)

if __name__ == '__main__':
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Add a simple guard to prevent multiple executions
    start_time = time.time()
    print(f"🔹 Starting training script at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load pretrained model if specified
    if args.pretrained_model is not None:
        print(f"🔹 Loading pretrained model from {args.pretrained_model}")
        # Find the best checkpoint in the pretrained model directory
        checkpoint_dir = os.path.join(args.pretrained_model, "best-checkpoint.ckpt")
        if os.path.exists(checkpoint_dir):
            try:
                model = LitRNNClassifier.load_from_checkpoint(checkpoint_dir)
                print("✅ Successfully loaded pretrained model")
                
                # Check if the model dimensions match the current dataset
                expected_input_dim = train_dataset[0][0].shape[1]
                expected_num_classes = train_dataset[0][1].shape[0]
                
                # Get the first layer of the GRU to check input dimension
                actual_input_dim = model.gru.input_size
                actual_num_classes = model.fc[-1].out_features
                
                if actual_input_dim != expected_input_dim:
                    print(f"⚠️ Warning: Model input dimension mismatch. Expected {expected_input_dim}, got {actual_input_dim}")
                if actual_num_classes != expected_num_classes:
                    print(f"⚠️ Warning: Model output dimension mismatch. Expected {expected_num_classes}, got {actual_num_classes}")
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not load pretrained model: {str(e)}")
                print("🔹 Continuing with a new model...")
        else:
            print(f"⚠️ Warning: No checkpoint found at {checkpoint_dir}")
            print("🔹 Continuing with a new model...")

    # Use user-provided learning rate without modification

    # Final safety check and training summary
    print(f"\n🔹 Training Configuration Summary:")
    print(f"   - Train dataset size: {len(train_dataset)}")
    print(f"   - Validation dataset size: {len(val_dataset)}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.lr}")
    print(f"   - Loss function: {args.loss_fn}")
    print(f"   - Max epochs: {args.epochs}")
    print(f"   - Device: {device}")
    print(f"   - Sanity check steps: {num_sanity_val_steps}")
    
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset is empty. Cannot proceed with training.")
    
    print(f"\n🚀 Starting training...")
    trainer.fit(model, train_loader, val_loader)
    print("✅ Training complete!")
    print(f"🔹 Total training time: {time.time() - start_time:.2f} seconds") 