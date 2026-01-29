import os
import torch as tr
import numpy as np
from torch import nn
from tqdm import tqdm
from src.basemodel_lora import BaseModelLoRA as BaseModel
from src.dataset import PFamDataset
from src.utils import load_config, predict
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
import csv

class FlattenLinear(nn.Module): # stackin perceptron
    """Flatten all predictions into a single vector and apply one Linear layer."""
    def __init__(self, num_models, num_classes, bias=True):
        super(FlattenLinear, self).__init__()
        self.linear = nn.Linear(num_models * num_classes, num_classes, bias=bias)

    def forward(self, x): # x: (num_models, batch, num_classes)
        # Permute to (batch, num_models, num_classes)
        x = x.permute(1, 0, 2)
        
        # Flatten to (batch, num_models * num_classes)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # Apply linear layer
        out = self.linear(x_flat)

        return out  # (batch, num_classes)

class FlattenMLP(nn.Module): # stacked MLP
    """Flatten all predictions and apply a two-layer MLP."""
    def __init__(self, num_models, num_classes, hidden_size=4096, bias=True):
        super(FlattenMLP, self).__init__()
        input_size = num_models * num_classes
        
        # Two-layer MLP
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x): # x: (num_models, batch, num_classes)
        # Permute to (batch, num_models, num_classes)
        x = x.permute(1, 0, 2)
        
        # Flatten to (batch, num_models * num_classes)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        
        # First layer with ReLU activation
        h = self.fc1(x_flat)  # (batch, hidden_size)
        h = tr.relu(h)
        
        # Second layer to output
        out = self.fc2(h)

        return out  # (batch, num_classes)
    

class EnsembleModel(nn.Module):
    def __init__(self, models_path, config, voting_strategy, ensemble_weights_path=None, 
                 exp_name=None):
        super(EnsembleModel, self).__init__()

        # Load model paths
        model_dirs = [os.path.join(models_path, d) for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        
        self.emb_path = config['emb_path']
        self.data_path = config['data_path']
        self.voting_strategy = voting_strategy
        self.path = models_path
        self.weights_file = None

        # Load categories
        cat_path = os.path.join(self.data_path, "categories.txt")
        with open(cat_path, 'r') as f:
            categories = [item.strip() for item in f]
        self.categories = categories

        # Store model directories and configs (defer loading to save GPU memory)
        self.model_dirs = []
        self.model_configs = []
        self.device = "cuda"

        # Sort model directories to ensure consistent order
        model_dirs.sort()

        # Load each model's configuration (but not weights yet)
        for model_dir in model_dirs:
            # Load the config.json to get the parameters
            config_path = os.path.join(model_dir, 'config.json')
            config = load_config(config_path)

            batch_size = config['batch_size']
            window_len = config['window_len']
            device = config['device']
            self.device = device

            self.model_dirs.append(model_dir)
            self.model_configs.append({
                'batch_size': batch_size,
                'window_len': window_len
            })
        
        # Initialize model weights based on voting strategy
        if self.voting_strategy in ['flatten_mlp', 'flatten_linear', 'weighted_model', 'weighted_families']:
            weights, weights_file = self._initialize_weights(model_dirs, 
                                                             ensemble_weights_path,
                                                             len(self.categories),
                                                             exp_name
                                                             )
            self.weights_file = weights_file
            self.model_weights = weights
            
    def _load_model(self, model_dir, config):
        """Load a single model from disk."""
        weights_path = os.path.join(model_dir, 'weights.pk')
        print("loading weights from", model_dir)
        model = BaseModel(len(self.categories), window_len=config['window_len'], device=self.device)
        model.load_state_dict(tr.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def fit(self, sequences_path, debug=False):
        if self.voting_strategy in ['flatten_mlp', 'flatten_linear', 'weighted_model', 'weighted_families']:
            # Save/load all_preds in the parent folder of model_dir
            preds_cache_path = os.path.join(self.path, 'all_preds_dev.pk')
            ref_cache_path = os.path.join(self.path, 'ref_dev.pk')

            if os.path.exists(preds_cache_path) and os.path.exists(ref_cache_path):
                print(f"Loading cached predictions from {preds_cache_path}")
                with open(preds_cache_path, 'rb') as f:
                    all_preds = pickle.load(f)
                with open(ref_cache_path, 'rb') as f:
                    ref = pickle.load(f)
            else:
                # Collect predictions from each model (load one at a time to save GPU memory)
                all_preds = []
                ref = None
                for i, model_dir in enumerate(self.model_dirs):
                    config = self.model_configs[i]
                    config["sequences"] = sequences_path
                    # Load model
                    net = self._load_model(model_dir, config)

                    dev_data = PFamDataset(
                        f"{self.data_path}dev.csv",
                        self.emb_path,
                        self.categories,
                        window_len=config['window_len'],
                        is_training=False, # to take centered window per domain
                        sequences=sequences_path,
                        debug=debug
                    )
                    dev_loader = tr.utils.data.DataLoader(dev_data, batch_size=config['batch_size'], num_workers=config.get("nworkers", 1))
                    print("predict model", i)
                    with tr.no_grad():
                        _, _, pred, ref, *_ = net.pred(dev_loader)
                        all_preds.append(pred.cpu())  # Move to CPU to free GPU memory

                    # Free GPU memory
                    del net
                    tr.cuda.empty_cache()

                # cache all_preds and ref for faster training later
                with open(preds_cache_path, 'wb') as f:
                    pickle.dump(all_preds, f)
                with open(ref_cache_path, 'wb') as f:
                    pickle.dump(ref, f)
            stacked_preds = tr.stack(all_preds).cuda()

        if self.voting_strategy == 'flatten_linear' or self.voting_strategy == 'flatten_mlp':
            save_log = True
            criterion = nn.CrossEntropyLoss()
            # Allow optional L2 regularization on voting-layer parameters
            optimizer = tr.optim.Adam(self.model_weights.parameters(), lr=0.01, weight_decay=1e-4)
            
            # Training log
            training_log = []
            log_file = self.weights_file.replace('.pt', '_training_log.csv')

            for epoch in tqdm(range(500), desc="Epochs"):
                pred_avg = self.model_weights(stacked_preds)
                loss = criterion(pred_avg.to(self.device), tr.argmax(ref, dim=1).to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if save_log:
                    pred_classes = tr.argmax(pred_avg, dim=1).cpu()
                    ref_classes = tr.argmax(ref, dim=1)
                    accuracy = (pred_classes == ref_classes).float().mean().item()
                    training_log.append({
                        'epoch': epoch + 1,
                        'loss': loss.item(),
                        'accuracy': accuracy
                    })
                    
                    # Save log and weights every epoch
                    with open(log_file, 'w', newline='') as csvfile:
                        fieldnames = ['epoch', 'loss', 'accuracy']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(training_log)
                    
                tr.save(self.model_weights.state_dict(), self.weights_file)

            print(f"Training completed. Final weights saved to {self.weights_file}")

        elif self.voting_strategy == 'weighted_model':
            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam([self.model_weights], lr=0.01)

            for epoch in tqdm(range(500), desc="Epochs"):
                pred_avg = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1).to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            tr.save(self.model_weights.detach().cpu(), self.weights_file)
            print(f"Saved model weights to {self.weights_file}")

        elif self.voting_strategy == 'weighted_families':
            criterion = nn.CrossEntropyLoss()
            optimizer = tr.optim.Adam([self.family_weights], lr=0.01)
            for epoch in tqdm(range(500), desc="Epochs"):
                pred_avg = tr.sum(stacked_preds * self.family_weights.view(len(self.model_dirs), 1, len(self.categories)), dim=0)
                loss = criterion(pred_avg, tr.argmax(ref, dim=1).to(self.device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            tr.save(self.family_weights.detach().cpu(), self.weights_file)
            print(f"Saved family weights to {self.weights_file}")

    def forward(self, batch):
        pred, _ = self.pred(batch)
        return pred

    def pred(self, partition='test', sequences=None, debug=False):
        # Predicts using the centered window method on the specified dataset.
        all_preds = []
        # Save/load all_preds in the parent folder of model_dir
        preds_cache_path = os.path.join(self.path, 'all_preds.pk' if partition=='test' else 'all_preds_dev.pk')
        if os.path.exists(preds_cache_path):
            print(f"Loading cached predictions from {preds_cache_path}")
            with open(preds_cache_path, 'rb') as f:
                all_preds = pickle.load(f) 
        else:
            for i, model_dir in enumerate(self.model_dirs):
                # Load the model's configuration and dataset
                config = self.model_configs[i]
                
                # Load model
                net = self._load_model(model_dir, config)

                test_data = PFamDataset(
                    f"{self.data_path}{partition}.csv",
                    self.emb_path,
                    self.categories,
                    window_len=config['window_len'],
                    is_training=False,
                    sequences=sequences,
                    debug=debug
                )
                test_loader = tr.utils.data.DataLoader(test_data,
                                                    batch_size=config['batch_size'],
                                                    num_workers=config.get("nworkers", 1))
                net_preds = []

                # Predict using the model
                with tr.no_grad():
                    test_loss, test_errate, pred, *_ = net.pred(test_loader)
                    net_preds.append(pred.cpu())  # Move to CPU to free GPU memory
                print(f"window_len = {config['window_len']}  - test_loss {test_loss:.5f} - test_errate {test_errate:.5f}")
                net_preds = tr.cat(net_preds)
                all_preds.append(net_preds)

                # Free GPU memory
                del net
                tr.cuda.empty_cache()
            # cache all_preds and ref for faster training later
            with open(preds_cache_path, 'wb') as f:
                pickle.dump(all_preds, f)

        stacked_preds = tr.stack(all_preds).to(self.device)
        preds, preds_bin = self._combine_ensemble_predictions(stacked_preds)
        return preds, preds_bin

    def pred_sliding(self, emb, step=4, use_softmax=False):
        all_preds = []
        all_centers = []

        # Save/load all_preds in the parent folder of model_dir
        preds_cache_path = os.path.join(self.path, 'all_preds_sliding.pk')
        if os.path.exists(preds_cache_path) and os.path.exists(ref_cache_path):
            print(f"Loading cached predictions from {preds_cache_path}")
            with open(preds_cache_path, 'rb') as f:
                all_preds, all_centers = pickle.load(f)
        else:        
            for i, model_dir in enumerate(self.model_dirs):
                config = self.model_configs[i]

                # Load model
                net = self._load_model(model_dir, config)

                net_preds = []
                centers, pred = predict(net, emb, config['window_len'],
                                        use_softmax=use_softmax, step=step)
                net_preds.append(pred.cpu())  # Move to CPU to free GPU memory
                all_preds.append(tr.cat(net_preds))
                all_centers.append(centers)

                # Free GPU memory
                del net
                tr.cuda.empty_cache()

            for c in all_centers:
                if not np.allclose(c, all_centers[0]):
                    raise ValueError("Model predictions have misaligned window centers.")

            # cache all_preds and ref for faster training later
            with open(preds_cache_path, 'wb') as f:
                pickle.dump((all_preds, all_centers), f)
            
        stacked_preds = tr.stack(all_preds).to(self.device)
        preds, preds_bin = self._combine_ensemble_predictions(stacked_preds)

        return centers, preds.cpu().detach()

    def pred_sliding_batch(self, sequences, pids, step=4, use_softmax=False):
        """
        Efficient batch prediction for multiple proteins.
        Loads each model once and processes all proteins before freeing.

        Args:
            sequences: Dict mapping pid -> embedding (or sequence data)
            pids: List of protein IDs to process
            step: Step size for sliding window
            use_softmax: Whether to apply softmax to predictions

        Returns:
            Dict mapping pid -> (centers, combined_preds)
        """
        # Store predictions per model per protein: {pid: [model_preds...]}
        all_model_preds = {pid: [] for pid in pids}
        all_centers = {pid: None for pid in pids}
        # Save/load all_preds in the parent folder of model_dir
        preds_cache_path = os.path.join(self.path, 'all_preds_sliding.pk')
        if os.path.exists(preds_cache_path):
            print(f"Loading cached predictions from {preds_cache_path}")
            with open(preds_cache_path, 'rb') as f:
                all_model_preds, all_centers = pickle.load(f)
        else:        
            # For each model, load once and process all proteins
            for i, model_dir in enumerate(self.model_dirs):
                config = self.model_configs[i]

                # Load model once
                net = self._load_model(model_dir, config)

                # Process all proteins with this model
                for pid in tqdm(pids, desc=f"Model {i+1}/{len(self.model_dirs)}"):
                    emb = sequences[pid]
                    centers, pred = predict(net, emb, config['window_len'],
                                            use_softmax=use_softmax, step=step)
                    all_model_preds[pid].append(pred.cpu())

                    # Store centers (should be same across models for same protein)
                    if all_centers[pid] is None:
                        all_centers[pid] = centers

                # Free GPU memory after processing all proteins with this model
                del net
                tr.cuda.empty_cache()

                # cache all_preds and ref for faster training later
                with open(preds_cache_path, 'wb') as f:
                    pickle.dump((all_model_preds, all_centers), f)


        # Combine predictions for each protein
        results = {}
        for pid in pids:
            stacked_preds = tr.stack(all_model_preds[pid]).to(self.device)
            preds, _ = self._combine_ensemble_predictions(stacked_preds)
            results[pid] = (all_centers[pid], preds.cpu().detach())

        return results

    def _initialize_weights(self, model_dirs, ensemble_weights_path, num_classes, exp_name=None):
        """Initializes the weights for the ensemble based on the voting strategy."""
        # Define the file name based on the voting strategy and experiment name (if provided)
        if exp_name:
            file_name = f"{self.voting_strategy}_ensemble_{exp_name}.pt"
        else:
            file_name = f"{self.voting_strategy}_ensemble.pt"
        # If ensemble_weights_path is provided, use it to load weights
        if ensemble_weights_path:
            weights_file = f"{ensemble_weights_path}{file_name}"
        else:
            weights_file = f"{self.path}{file_name}"

        if self.voting_strategy == "flatten_linear":
            weights = FlattenLinear(len(model_dirs), num_classes).to(self.device)
            weights.load_state_dict(tr.load(weights_file, map_location=self.device)) 
            return weights, weights_file
        elif self.voting_strategy == "flatten_mlp":
            weights = FlattenMLP(len(model_dirs), num_classes).to(self.device)
            weights.load_state_dict(tr.load(weights_file, map_location=self.device)) 
            return weights, weights_file

        elif self.voting_strategy == 'weighted_model':
            if ensemble_weights_path and os.path.exists(weights_file):
                weights = nn.Parameter(tr.load(weights_file)).to(self.device)
                print(f"Loaded model weights from {weights_file}")
            else:
                weights = nn.Parameter(tr.rand(len(model_dirs), device=self.device))
                if ensemble_weights_path:
                    print(f"Warning: {weights_file} not found, using random init.")
            return weights, weights_file

        elif self.voting_strategy == 'weighted_families':
            if ensemble_weights_path and os.path.exists(weights_file):
                weights = nn.Parameter(tr.load(weights_file).to(self.device))
                print(f"Loaded family weights from {weights_file}")
            else:
                weights = nn.Parameter(tr.rand(len(model_dirs), len(self.categories), device=self.device))
                if ensemble_weights_path:
                    print(f"Warning: {weights_file} not found, using random init.")
            return weights, weights_file
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")

    def _combine_ensemble_predictions(self, stacked_preds):
        """ Combines predictions from the ensemble models based on the voting strategy."""
        
        if self.voting_strategy == 'flatten_linear' or self.voting_strategy == 'flatten_mlp':
            pred = self.model_weights(stacked_preds)
            pred_bin = tr.argmax(pred, dim=1)
        elif self.voting_strategy == 'score_voting':
            pred = tr.mean(stacked_preds, dim=0)
            pred_bin = tr.argmax(pred, dim=1)
        elif self.voting_strategy == 'weighted_model':
            pred = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'weighted_families':
            pred = tr.sum(stacked_preds * self.model_weights.view(len(self.model_dirs), 1, len(self.categories)), dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'simple_voting':
            pred_classes = tr.mode(tr.argmax(stacked_preds, dim=2), dim=0)[0]
            pred = tr.nn.functional.one_hot(pred_classes, num_classes=len(self.categories)).float()
            pred_bin = tr.argmax(pred, dim=1)

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        return pred, pred_bin
