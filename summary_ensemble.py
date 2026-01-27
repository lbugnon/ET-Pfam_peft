# load train log from all output directories and summarize results

import os
import pandas as pd

# Path to ensemble outputs
ensemble_dir = 'models_lora_finetuned_mini'
summary = []

for output_dir in os.listdir(ensemble_dir):
    full_output_path = os.path.join(ensemble_dir, output_dir)
    if not output_dir.startswith('output_') or not os.path.isdir(full_output_path):
        continue

    centered_error = None
    test_log_path = os.path.join(full_output_path, 'centered_test.csv')
    if os.path.exists(test_log_path):
        df = pd.read_csv(test_log_path)
        centered_error = df["Error Rate (%)"].item()
    
    test_log_path = os.path.join(full_output_path, 'sliding_window_test.csv')
    swm, swa, swc = None, None, None
    if os.path.exists(test_log_path):
        df = pd.read_csv(test_log_path)
        swm = df[df.Score == "Max"]["Error Rate (%)"].item()
        swa = df[df.Score == "Area"]["Error Rate (%)"].item()
        swc = df[df.Score == "Coverage"]["Error Rate (%)"].item()
    
    summary.append({
        'Output Directory': output_dir,
        'centered_error': centered_error,
        'swm': swm,
        'swa': swa,
        'swc': swc,
    })

summary_df = pd.DataFrame(summary)
print(summary_df)