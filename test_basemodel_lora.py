"""
This script is designed to test an individual base model, using the centered
and sliding window techniques.
Parameters:
    -o, --output_path: Path containing the pre-trained model weights to be
                        evaluated, and where the results will be saved.
Usage example:
    python3 test_basemodel.py -o models/mini/model1/
"""
import os 
import argparse
import torch as tr
from Bio import SeqIO

#from src.basemodel import BaseModel
from src.basemodel_lora import BaseModelLoRA as BaseModel

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

    categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]

    # Load the trained model
    model = BaseModel(len(categories), window_len=config['window_len'], device=config['device'])
    
    state_dict = tr.load(f"{output_path}/weights.pk", map_location="cuda:0")
    # ====
    # Load full pretrained
    model.load_state_dict(state_dict)
    
    # Fix to load pretrained model without esm weights
    #model.cnn.load_state_dict({k.replace("cnn.", ""): state_dict[k] for k in state_dict if "cnn" in k})
    #model.fc.load_state_dict({k.replace("fc.", ""): state_dict[k] for k in state_dict if "fc" in k})
    # ====

    model.eval()

    # Centered window test
    print("Testing centered window...")
    centered_window_test(config, model, output_path)

    # Sliding window test
    print("Testing sliding window...")
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(f"{config['data_path']}test.fasta", "fasta")}
    sliding_window_test(config, model, output_path, sequences=sequences)
