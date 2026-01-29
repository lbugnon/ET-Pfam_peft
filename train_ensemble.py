"""
This script is used to train ensemble models based on the specified voting strategy.
Parameters:
    -v, --voting_strategy: Voting strategy to use (e.g., 'weighted_model', 
                            'weighted_families', 'all').
    -m, --models_path: Path to the directory containing the models to ensemble.
    -c, --config_path: Path to the configuration file (.json).
    -e, --exp_name: Experiment name for saving ensemble weights.
Usage example:
    python3 train_ensemble.py -v all -m models/mini/ 
"""
import os
import argparse
import torch as tr
from src.ensemble import EnsembleModel
from src.utils import load_config

tr.multiprocessing.set_sharing_strategy('file_system')

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--voting_strategy", type=str, required=True, 
                        help="Voting strategy: 'flatten_linear', 'weighted_model'," \
                        " 'weighted_families', 'all'")
    parser.add_argument("-m","--models_path", type=str, required=True, 
                        help="Path to the models to ensemble",
                        default="models/mini/")
    
    parser.add_argument("-c","--config_path", type=str, required=False,
                        help="Path to the config file (.json)",
                        default="config/base.json")
    parser.add_argument("-e", "--exp_name", type=str, required=False,
                        help="Experiment name for saving ensemble weights",
                        default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: limit validation to 100 sequences for faster iteration.") 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser()
    width = os.get_terminal_size().columns

    valid_strategies = ['flatten_mlp', 'flatten_linear', 'weighted_model', 'weighted_families', 'all']
    if args.voting_strategy not in valid_strategies:
        if args.voting_strategy in ['simple_voting', 'score_voting']:
            raise ValueError(f"Voting strategy '{args.voting_strategy}' does not require training. " \
                             f"Use 'weighted_model', 'weighted_families', or 'all' to train ensemble models.")
        else:
            raise ValueError(f"Invalid voting strategy: {args.voting_strategy}. " \
                             f"Use 'weighted_model', 'weighted_families', or 'all' to train ensemble models.")

    config = load_config(args.config_path)

    if args.voting_strategy == 'all':
        strategies_to_test = valid_strategies[:-1] # Exclude 'all' 
    else:
        strategies_to_test = [args.voting_strategy]
    
    for strategy in strategies_to_test:
        print("\n" + ">" * width)
        print("Fitting ensemble model with voting strategy: ", strategy)

        ensemble = EnsembleModel(args.models_path, config, strategy, 
                                 exp_name=args.exp_name)
        ensemble.fit(sequences_path=f"{config['data_path']}dev.fasta", debug=args.debug)