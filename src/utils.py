import os
import json
import pandas as pd
import numpy as np
import torch as tr
from torch.nn.functional import softmax

def predict(net, seq, window_len, use_softmax=True, step=8):
    """
    Predicts using a sliding window on the given sequence.
    Args:
        net: BaseModel or BaseModelLoRA.
        seq: The input sequence (string).
        window_len: The length of the sliding window.
        use_softmax: Whether to apply softmax to the predictions.
        step: Step size for the sliding window.
    Returns:
        centers: The center positions of the sliding windows.
        pred: The predictions from the model (num_windows x num_classes).
    """
    L = len(seq)
    centers = np.arange(0, L, step)
    
    # Compute embeddings once for the entire sequence
    with tr.no_grad():
        emb = net.compute_embeddings(seq)
    
    predictions = []
    with tr.no_grad():
        for center in centers:
            start_pos = max(0, center - window_len // 2)
            end_pos = min(L, start_pos + window_len)
            # Adjust if we're at the end of the sequence
            if end_pos - start_pos < window_len:
                start_pos = max(0, end_pos - window_len)
            
            # Use pre-computed embeddings for efficiency
            pred = net.forward_from_embeddings(emb, [start_pos], [end_pos]).cpu().detach()
            predictions.append(pred)
    
    pred = tr.cat(predictions, dim=0)
    
    if use_softmax:
        pred = softmax(pred, dim=1)

    return centers, pred

def load_config(path='config/base.json'):
    """
    Loads a model configuration and merges it with environment-specific settings.
    Args:
        path (str): Path to the model config JSON file.
    Returns:
        dict: Combined configuration dictionary.
    """
    # Load model config from given path
    with open(path, 'r') as f:
        model = json.load(f)
    
    # Load env config from default path
    with open('config/env.json', 'r') as f:
        env = json.load(f)

    # Initialize config with model settings
    config = {**model}

    # Add environment-specific settings 
    keys_to_add = ['nworkers', 'device', 'emb_path', 'continue_training']
    for key in keys_to_add: 
        config[key] = env[key]
        
    # Add the path to the datase
    if config['dataset'] in ["full", "mini"]:
        config['data_path'] = env[f'{model["dataset"]}_path']
    else:
        raise ValueError(f"Invalid dataset name: {model['dataset']}. Expected 'full' or 'mini'.")

    return config

class ResultsTable():
    """Save results in a DataFrame and export to CSV."""
    
    def __init__(self, is_ensemble=False):
        """Initializes the ResultsTable"""
        self.is_ensemble = is_ensemble
        self.label_name = "Model" if not is_ensemble else "Strategy"
        self.df = pd.DataFrame(columns=[self.label_name, "CwS", "SwA", "SwC"])

    def add_entry(self, label, cws, swa, swc):
        """Add a new entry to the results DataFrame"""
        new_row = {
            self.label_name: label,
            "CwS": round(cws, 2),
            "SwA": round(swa, 2),
            "SwC": round(swc, 2)
        }
        self.df.loc[len(self.df)] = new_row

    def save(self, filepath):
        """Save the results DataFrame to a CSV file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.df.to_csv(filepath, index=False)
