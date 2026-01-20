# load train log from all output directories and summarize results
import os
import pandas as pd
summary = []
for output_dir in os.listdir('.'):
    if "output_" not in output_dir or not os.path.isdir(output_dir):
        continue

    log_path = os.path.join(output_dir, 'train_summary.csv')
    if not os.path.exists(log_path):
        continue
    df = pd.read_csv(log_path)
    if len(df) == 0:
        continue
    best_error = df['Best error'].iloc[-1]
    ep = df['Ep'].iloc[-1]
    summary.append({
        'Output Directory': output_dir,
        'Best error': best_error,
        'Final Epoch': ep
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(by='Best error')
print(summary_df)