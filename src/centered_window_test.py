import os
import numpy as np
import pandas as pd
import torch as tr
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from src.dataset import PFamDataset

def centered_window_test(config, model, output_folder, is_ensemble=False,
                         voting_strategy=None, partition='test', debug=False):
    """
    Evaluates a model (individual or ensemble) on the test dataset using the 
    centered window technique.
    Args:
        config (dict): Configuration dictionary
        model (torch.nn.Module): Single model or ensemble.
        output_folder (str): Folder where results will be saved.
        is_ensemble (bool, optional): Whether the model is an ensemble. Default is False.
        voting_strategy (str, optional): Voting strategy used (e.g., "simple voting").
        debug (bool, optional): Reduce the number of samples for a quick test (default: False).     
    """
    # Load configuration parameters
    data_path = config['data_path']
    emb_path = config['emb_path']
    cat_path = os.path.join(data_path, "categories.txt")
    categories = [line.strip() for line in open(cat_path)]

    # Set default parameters for ensemble or individual model
    if is_ensemble:
        window_len, batch_size = 128, 128  # Default values for ensembles
    else:
        window_len, batch_size = config.get("window_len", 32), config.get("batch_size", 32)

    print(window_len, batch_size, config["sequences"])
    sequences = config.get("sequences", None)
    data = PFamDataset(f"{data_path}{partition}.csv", emb_path, categories,
                      window_len=window_len, is_training=False, sequences=sequences, debug=debug)
    loader = DataLoader(data,
                       batch_size=batch_size,
                       num_workers=config.get("nworkers", 4))

    model.eval()
    if is_ensemble:
        # Get the ensemble predictions
        _, pred_bin = model.pred(partition=partition, sequences=sequences, debug=debug)

        # Collect ground truth references
        ref = [] 
        for _, y, *_ in tqdm(loader, desc="Collecting ground truth"):
            ref.append(y.cpu())
        ref = tr.cat(ref)
        ref_bin = tr.argmax(ref, dim=1)

        # Calculate metrics
        accuracy = accuracy_score(ref_bin.cpu().numpy(), pred_bin.cpu().numpy())
        err_rate = 1 - accuracy

    else: 
        # Get the base model predictions
        _, err_rate, pred, ref, *_ = model.pred(loader, debug=debug)

        ref_bin = tr.argmax(ref, dim=1)
        pred_bin = tr.argmax(pred, dim=1)
     
    total_errors = int(np.sum(1 - (pred_bin.cpu().numpy() == ref_bin.cpu().numpy())))

    # Print results
    print(f"Error rate:   {err_rate * 100:.2f}% ({total_errors}/{len(ref_bin)})")
    if voting_strategy:
        print(f"Strategy:     {voting_strategy}")

    # Save stats to CSV
    stats_file = os.path.join(output_folder, f"centered_{partition}.csv")
    stats = {
        "Error Rate (%)": [f"{err_rate * 100:.2f}"],
        "Total Errors": [f"{total_errors}/{len(ref_bin)}"]
    }
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(stats_file, index=False)
    
    return err_rate * 100