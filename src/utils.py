import os
import json
import pandas as pd
import numpy as np
import torch as tr
from torch.nn.functional import softmax

def predict(net, emb, window_len, use_softmax=True, step=8):
    """
    Predicts using a sliding window on the given embeddings.
    Args:
        net: BaseModel or Ensemble Model.
        emb: The input embeddings of shape (batch_size, sequence_length).
        window_len: The length of the sliding window.
        use_softmax: Whether to apply softmax to the predictions.
        step: Step size for the sliding window.
    Returns:
        centers: The center positions of the sliding windows.
        pred: The predictions from the model.
    """
    L = emb.shape[1] if type(emb) is tr.Tensor else len(emb)
    B = emb.shape[0] if type(emb) is tr.Tensor else 1
    centers = np.arange(0, L, step)
    
    with tr.no_grad():
        pred = net(seq, start, end).cpu().detach()
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
