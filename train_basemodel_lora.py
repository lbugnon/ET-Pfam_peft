"""
Train base model with LoRA support and optional pretrained weights.

This script supports multiple training modes:
1. Train everything from scratch (ESM2 LoRA + CNN + FC)
2. Load pretrained CNN/FC and continue training everything
3. Load pretrained CNN/FC, freeze them, and only train ESM2 with LoRA (fastest)

Parameters:
    -o, --output_path: Path to save the model.
    -c, --config_path: Path to the configuration file (json).
    -p, --pretrained_path: (Optional) Path to pretrained model weights (weights.pk).
    --freeze_cnn: (Optional) Freeze CNN parameters after loading.
    --freeze_fc: (Optional) Freeze FC parameters after loading.
    --freeze_cnn_fc: (Optional) Freeze both CNN and FC parameters after loading.

Usage examples:
    # Train from scratch
    python3 train_basemodel_lora.py -o output/ -c config/lora_finetune.json
    
    # Load pretrained CNN/FC and continue training all
    python3 train_basemodel_lora.py -o output/ -c config/lora_finetune.json -p output_full_seq_lorav2/weights.pk
    
    # Load pretrained CNN/FC, freeze them, train only ESM2 LoRA (quickest)
    python3 train_basemodel_lora.py -o output_esm2_only/ -p output_full_seq_lorav2/weights.pk --freeze_cnn_fc
"""
import os 
import argparse
import shutil
from src.utils import load_config
from src.train import train

def parser():
    parser = argparse.ArgumentParser(description="Train a base model with LoRA.")
    parser.add_argument("-o", "--output_path", type=str, required=True, 
                        help="Path to save the model.")    
    parser.add_argument("-c", "--config_path", type=str, required=False, 
                        help="Path to the configuration file (JSON).",
                        default="config/lora_finetune.json")
    parser.add_argument("-p", "--pretrained_path", type=str, required=False,
                        help="Path to pretrained model weights (weights.pk). If provided, CNN/FC weights will be loaded.",
                        default=None)
    parser.add_argument("--freeze_cnn", action="store_true",
                        help="Freeze CNN parameters (requires --pretrained_path).")
    parser.add_argument("--freeze_fc", action="store_true",
                        help="Freeze FC parameters (requires --pretrained_path).")
    parser.add_argument("--freeze_cnn_fc", action="store_true",
                        help="Freeze both CNN and FC parameters (requires --pretrained_path).")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: limit validation to 100 sequences for faster iteration.")

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # Validate freeze options
    if (args.freeze_cnn or args.freeze_fc or args.freeze_cnn_fc) and not args.pretrained_path:
        parser.error("--freeze_cnn, --freeze_fc, and --freeze_cnn_fc require --pretrained_path to be specified.")
    
    return args

if __name__ == "__main__":
    args = parser()
    config_path = args.config_path
    output_path = args.output_path

    # Copy the config file to the output path
    shutil.copyfile(config_path, os.path.join(output_path, "config.json"))

    # Load the configuration
    config = load_config(config_path)
    
    # Add pretrained and freeze options to config
    config['pretrained_path'] = args.pretrained_path
    config['freeze_cnn'] = args.freeze_cnn 
    config['freeze_fc'] = args.freeze_fc 
    config['freeze_cnn_fc'] = args.freeze_cnn_fc
    config['debug'] = args.debug 
     
    categories = [line.strip() for line in open(f"{config['data_path']}categories.txt")]

    # Train the model 
    if args.pretrained_path:
        freeze_msg = []
        if config['freeze_cnn']:
            freeze_msg.append("CNN")
        if config['freeze_fc']:
            freeze_msg.append("FC")
        if freeze_msg:
            print(f"Training with pretrained weights from {args.pretrained_path} (freezing: {', '.join(freeze_msg)})...")
        else:
            print(f"Training with pretrained weights from {args.pretrained_path} (all parameters trainable)...")
    else:
        print("Training from scratch...")
    
    train(config, categories, output_path)
