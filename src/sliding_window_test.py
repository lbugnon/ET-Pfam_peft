import os
import pickle
import torch as tr
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils import predict

def _predict_max(categories, pred, start, end, gold_label): 
    """Prediction using the maximum peak value in the interval."""
    
    # Select the class with the maximum peak value in the interval
    pred_class = np.argmax(pred[start:end, :].max(axis=0).values)

    # Get the maximum score for that class in the interval
    score = tr.max(pred[start:end, pred_class])
    prediction = categories[pred_class]

    # Return predicted label, confidence score, and if the prediction is correct
    return [prediction, score.item(), prediction == gold_label]

def _predict_area(categories, pred, start, end, gold_label):
    """Prediction using the "area" under the curve."""

    # Select the class with the largest accumulated score across the interval
    prediction = categories[np.argmax(pred[start:end].sum(axis=0))]

    # Score as the "area" under the curve of the predicted class
    score = pred[start:end].sum(axis=0).max()

    # Return predicted label, confidence score, and if the prediction is correct
    return [prediction, score.item(), prediction == gold_label]

def _predict_coverage(categories, pred, start, end, gold_label):
    """Prediction using the most common class along the domain."""

    # Find the most frequently predicted class across the interval
    candidates = np.argmax(pred[start:end], axis=1)
    counts = np.bincount(candidates)
    prediction = categories[np.argmax(counts)]

    # Score reflects class dominance (frequency ratio in the interval)
    score = counts.max() / len(candidates)

    # Return predicted label, confidence score, and if the prediction is correct
    return [prediction, score.item(), prediction == gold_label]


def sliding_window_test(config, model, output_path, is_ensemble=False, partition='test', sequences=None, debug=False):
    """
    Run sliding window prediction on a test dataset and evaluate three prediction strategies:
    max score, area under curve, and coverage-based majority voting.
    Args:
        config (dict): Configuration dictionary containing paths and parameters.
        model (torch.nn.Module): The trained model or ensemble to evaluate.
        output_path (str): Directory to save the results and summary.
        debug (bool, optional): Reduce the number of samples for a quick test (default: False).
    """
    # Paths and parameters
    data_path = config['data_path']
    cat_path = os.path.join(data_path,"categories.txt")
    categories = [line.strip() for line in open(cat_path)]

    # Load the test dataset
    dataset = pd.read_csv(f"{config['data_path']}{partition}.csv")
    if debug:
        dataset = dataset.groupby('PID').head(5).reset_index(drop=True)

    if sequences is None:
        fasta_path = config.get("sequences", None)
        if isinstance(fasta_path, str):
            from Bio import SeqIO
            sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}
    # print(f"Total rows: {len(dataset)}")

    errors = []
    nOKs, nOKsArea, nOKsCoverage = 0, 0, 0

    model.eval() # Set model to evaluation mode

    # Get unique PIDs
    pids = dataset.PID.unique().tolist()
    if debug:
        pids = pids[:5]

    # For ensemble models, use batch prediction (loads each model once for all proteins)
    if is_ensemble:
        print("Running batch prediction for ensemble (efficient mode)...")
        batch_results = model.pred_sliding_batch(sequences, pids,
                                                  step=config['step'],
                                                  use_softmax=config['soft_max'])

    # Iterate over the proteins to process predictions
    for pid in tqdm(pids):

        # Get the predictions from the model using the sliding window approach
        if is_ensemble:
            # For ensemble models, use precomputed batch results
            centers, pred = batch_results[pid]
        else:
            # For single models, use the predict function
            emb = sequences[pid]
            centers, pred = predict(model, emb, config['window_len'],
                                    use_softmax=config['soft_max'], step=config['step'])

        # Get labeled domains for the current PID
        ref = dataset[dataset.PID == pid].sort_values(by="start")
        if ref.empty:
            print(f"Missing reference: {pid}")
            continue

        for _, row in ref.iterrows():
            start, end = row.start, row.end

            if end-start < 10:
                print(f"Short domain: {pid}")
                continue

            # Determine the domain's start and end indices in the prediction array
            pred_start = np.argmin(np.abs(centers-start))
            pred_end = np.argmin(np.abs(centers-end))

            if (pred_start<pred_end):
                summary = [pid, start, end, row.PF]

                # Prediction using the maximum peak value in the interval
                results_max = _predict_max(categories, pred, pred_start, pred_end, row.PF)
                summary += results_max
                nOKs += int(results_max[2])
                        
                # Prediction using the "area" under the curve
                results_area = _predict_area(categories, pred, pred_start, pred_end, row.PF)
                nOKsArea += int(results_area[2])
                summary += results_area
                
                # Prediction using the most common class along the domain
                results_coverage = _predict_coverage(categories, pred, pred_start, pred_end, row.PF)
                nOKsCoverage += int(results_coverage[2])
                summary += results_coverage
                
                errors.append(summary)
                            
            else:
                print(f"Missing index: {pid}")

    # Save errors to CSV
    cols = [   
    "PID", "start", "end", "PF",
    "pred_max", "score_max", "pred_ok_max",
    "pred_area", "score_area", "score_ok_area",
    "pred_coverage", "score_coverage", "score_ok_coverage"
    ]

    errors_file = os.path.join(output_path, "errors.csv")
    errors = pd.DataFrame(errors, columns=cols)
    errors.to_csv(f"{errors_file}", index=False)

    # Calculate error rates
    n_total = len(errors)
    n_errors_max = n_total - nOKs
    n_errors_area = n_total - nOKsArea
    n_errors_coverage = n_total - nOKsCoverage

    perc_error_max = (n_errors_max / n_total) * 100
    perc_error_area = (n_errors_area / n_total) * 100
    perc_error_coverage = (n_errors_coverage / n_total) * 100

    # Print error statistics
    print(f"Error rate Max:      {(perc_error_max):6.2f}% ({n_errors_max}/{n_total})")
    print(f"Error rate Area:     {(perc_error_area):6.2f}% ({n_errors_area}/{n_total})")
    print(f"Error rate Coverage: {(perc_error_coverage):6.2f}% ({n_errors_coverage}/{n_total})")

    # Save stats to CSV
    stats = {
        "Score": ["Max", "Area", "Coverage"],
        "Error Rate (%)": [
            f"{perc_error_max:.2f}",
            f"{perc_error_area:.2f}",
            f"{perc_error_coverage:.2f}"
        ],
        "Total Errors": [
            f"{n_errors_max}/{n_total}",
            f"{n_errors_area}/{n_total}",
            f"{n_errors_coverage}/{n_total}"
        ]
    }

    stats_file = os.path.join(output_path, f"sliding_window_{partition}.csv")
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(stats_file, index=False)

    return perc_error_max, perc_error_area, perc_error_coverage