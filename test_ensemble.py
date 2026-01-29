"""
This script runs ensemble evaluation on a set of models using various voting 
strategies. It performs both centered window and sliding window testing, and 
saves the results into a CSV file.
Parameters:
    -v, --voting_strategy: Voting strategy to use (e.g., 'simple_voting', 
                            'score_voting', 'weighted_model', 'weighted_families', 
                            'all').
    -m, --models_path: Path to the directory containing the models to ensemble.
    -c, --config_path: Path to the configuration file (.json).
    -w, --ensemble_weights_path: Path to the model weights for weighted 
                                    voting strategies.
    -o, --output_path: Path to save the results.
    -e, --exp_name: Experiment name for saving ensemble weights.
    -p, --partition: Dataset partition to test on (default: 'test').
Usage example:
    python3 test_ensemble.py -v all -m models/mini/
"""
import os
import argparse
import torch as tr
from src.ensemble import EnsembleModel
from src.centered_window_test import centered_window_test
from src.sliding_window_test import sliding_window_test
from src.utils import load_config, ResultsTable

tr.multiprocessing.set_sharing_strategy('file_system')

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--voting_strategy", type=str, required=True, 
                        help="Voting strategy: 'flatten_linear', 'weighted_families', 'simple_voting', 'score_voting', " \
                        "'weighted_model', 'all'")
    parser.add_argument("-m","--models_path", type=str, required=True, 
                        help="Path to the models to ensemble",
                        default="models/mini/")
    
    parser.add_argument("-c","--config_path", type=str, required=False,
                        help="Path to the config file (.json)",
                        default="config/base.json")
    parser.add_argument("-w", "--ensemble_weights_path", type=str, required=False,
                        help="Path (folder) to the model weights for weighted voting strategies")
    parser.add_argument("-p", "--partition", type=str, required=False,
                        help="Dataset partition to test on (default: 'test')",
                        default='test')     
    parser.add_argument("-o", "--output_path", type=str, required=False,
                        help="Path to save the results",
                        default=None)
    parser.add_argument("-e", "--exp_name", type=str, required=False,
                        help="Experiment name for saving ensemble weights",
                        default=None)
    parser.add_argument("--debug", action='store_true',
                        help="Reduce the number of samples for a quick test")
    
    args = parser.parse_args()
    return args

def run_ensemble_tests(models_path, config, voting_strategy, output_path,
                       ensemble_weights_path, exp_name=None, partition='test', debug=False):
    """
    Loads the ensemble weights (if needed) and runs both sliding and centered 
    window tests.
    Args:
        models_path (str): Path to ensemble model directories.
        config (dict): Configuration dictionary.
        voting_strategy (str): Selected voting strategy.
        output_path (str): Directory to store results.
        ensemble_weights_path (str, optional): Path for ensemble weights.
        exp_name (str, optional): Experiment name for saving ensemble weights.
        partition (str, optional): Dataset partition to test on (default: 'test').
        debug (bool, optional): Reduce the number of samples for a quick test (default: False).
    """
    width = os.get_terminal_size().columns

    config["sequences"] = f"data/mini/{partition}.fasta"
    print(f"Running ensemble tests with voting strategy: {voting_strategy}")
    if voting_strategy in ['flatten_linear', 'weighted_model', 'weighted_families']:
        print(f"Using model weights from: {ensemble_weights_path}")

    ensemble = EnsembleModel(models_path, config,
                             voting_strategy, 
                             ensemble_weights_path=ensemble_weights_path, 
                             exp_name=exp_name)

    # Test the ensemble with centered and sliding window methods
    print("\n" + "-" * width)
    print("\nRunning centered window test...")
    CwS = centered_window_test(config, ensemble, output_path, is_ensemble=True,
                         voting_strategy=voting_strategy, partition=partition, debug=debug)

    print("\n" + "-" * width)
    print("\nRunning sliding window test...")
    _, SwA, SwC = sliding_window_test(config, ensemble, output_path, is_ensemble=True,
                                       partition=partition, debug=debug)

    return CwS, SwA, SwC

if __name__ == "__main__":
    args = parser()
    width = os.get_terminal_size().columns

    # Validate voting strategy
    valid_strategies = ['flatten_linear', 'weighted_families', 'simple_voting', 'score_voting', 'weighted_model',
                         'all']
    if args.voting_strategy not in valid_strategies:
        raise ValueError(f"Invalid voting strategy: {args.voting_strategy}. " \
                         f"Choose from {valid_strategies[:-1]} or 'all'.")

    config = load_config(args.config_path)
    if config["dataset"] == "mini":
        config["sequences"] = "data/mini/test.fasta"

    # Set output path
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = f'results/{config["dataset"]}/'
    os.makedirs(output_path, exist_ok=True)
    
    # Determine strategies to test
    if args.voting_strategy == 'all':
        strategies_to_test = valid_strategies[:-1] # Exclude 'all' 
    else:
        strategies_to_test = [args.voting_strategy]

    # Set ensemble weights path. If not provided, use models_path
    if args.ensemble_weights_path:
        ensemble_weights_path = args.ensemble_weights_path
    else:
        ensemble_weights_path = args.models_path

    print("\n" + ">" * width)
    results = ResultsTable(is_ensemble=True)

    for strategy in strategies_to_test:  
        path = os.path.join(output_path, strategy)
        os.makedirs(path, exist_ok=True)

        CwS, SwA, SwC = run_ensemble_tests(models_path=args.models_path, 
                                            config=config,
                                            voting_strategy=strategy,
                                            output_path=path,
                                            ensemble_weights_path=ensemble_weights_path,
                                            exp_name=args.exp_name,
                                            partition=args.partition,
                                            debug=args.debug)
        results.add_entry(strategy, CwS, SwA, SwC)

    results_file = os.path.join(output_path, "ensemble_metrics.csv")
    results.save(results_file)
