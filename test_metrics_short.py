#!/usr/bin/env python3
"""
Short test script to verify metrics logging works correctly
"""

import subprocess
import time
import os

def main():
    print("🧪 Testing metrics logging with a short training run...")
    
    # Create a test directory
    test_dir = "test_metrics_short"
    
    # Run a short training session
    cmd = [
        "python", "run_experiment_gru_lightning.py",
        "--save_dir", test_dir,
        "--epochs", "5",  # Very short run
        "--eval_interval", "1",  # Evaluate every epoch
        "--log_interval", "1",  # Log every step
        "--lr", "1e-3",
        "--batch_size", "32",  # Smaller batch size for testing
        "--use_gpu",
        "--test_size", "0.2",  # Larger test size for more validation data
        "--dropout", "0.1",
        "--loss_fn", "bce",
        "--num_workers", "1"
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        # Run the training
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        end_time = time.time()
        
        print(f"⏱️  Training completed in {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            
            # Check the output for metrics logging
            output = result.stdout + result.stderr
            
            # Look for metrics logging messages
            if "val_f1=" in output:
                print("✅ Validation F1 metrics found in output")
            else:
                print("❌ Validation F1 metrics not found in output")
                
            if "train_f1=" in output:
                print("✅ Training F1 metrics found in output")
            else:
                print("❌ Training F1 metrics not found in output")
            
            # Check if metrics file was created
            metrics_file = os.path.join(test_dir, "metrics", "metrics.csv")
            if os.path.exists(metrics_file):
                print(f"✅ Metrics file created: {metrics_file}")
                
                # Read and check the metrics
                import pandas as pd
                df = pd.read_csv(metrics_file)
                
                expected_columns = ['val_f1', 'val_map', 'val_auc', 'train_f1_eval', 'train_map_eval', 'train_auc_eval']
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if missing_columns:
                    print(f"❌ Missing columns: {missing_columns}")
                else:
                    print("✅ All expected metric columns present!")
                    
                # Check for non-NaN values
                for col in expected_columns:
                    if col in df.columns:
                        non_nan_count = df[col].notna().sum()
                        print(f"  {col}: {non_nan_count}/{len(df)} non-NaN values")
                
            else:
                print(f"❌ Metrics file not found: {metrics_file}")
                
        else:
            print(f"❌ Training failed with return code: {result.returncode}")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running training: {str(e)}")
    
    # Clean up test directory
    if os.path.exists(test_dir):
        print(f"🧹 Cleaning up test directory: {test_dir}")
        import shutil
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    main() 