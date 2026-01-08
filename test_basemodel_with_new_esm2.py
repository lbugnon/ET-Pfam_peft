"""
This script is designed to test an individual base model with ESM2 (without LoRA), 
using the centered and sliding window techniques.
Parameters:
    -o, --output_path: Path containing the pre-trained model weights to be
                        evaluated, and where the results will be saved.
Usage example:
    python3 test_basemodel_with_new_esm2.py -o models/mini/model1/
"""
import os 
import argparse
import torch as tr
from Bio import SeqIO

from src.basemodel_lora import BaseModelLoRA
from src.utils import load_config
from src.centered_window_test import centered_window_test
from src.sliding_window_test import sliding_window_test

def parser():
    parser = argparse.ArgumentParser(description="Test a base model.")
    parser.add_argument("-o", "--output_path", type=str, required=True, 
                        help="Path containing the pre-trained model weights to " \
                        "be evaluated, and where the results will be saved")    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser()
    output_path = args.output_path
    config_path = os.path.join(output_path, "config.json")

    # Load the configuration
    config = load_config(config_path)
    config["emb_path"] = None
    config["sequences"] = f"{config['data_path']}test.fasta"
    print(config["sequences"])
    categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]

    # Load the model with ESM2 (no LoRA)
    model = BaseModelLoRA(len(categories), lr=config['lr'], device=config['device'], use_lora=False)
    
    state_dict = tr.load(f"{output_path}/weights.pk", map_location=config['device'])
    
    # Load only CNN and FC weights (ESM2 uses default pretrained weights)
    # Filter to load only the CNN and FC parts, excluding ESM2 weights
    model_state = {k: v for k, v in state_dict.items() if k.startswith("cnn.") or k.startswith("fc.")}
    model.load_state_dict(model_state, strict=False)
    
    model.eval()

    # Centered window test
    print("Testing centered window...")
    centered_window_test(config, model, output_path)

    # Sliding window test
    print("Testing sliding window...")
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(f"{config['data_path']}test.fasta", "fasta")}
    sliding_window_test(config, model, output_path, sequences=sequences)


