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
        if self.voting_strategy in ['weighted_model', 'weighted_families']:
            weights, weights_file = self._initialize_weights(model_dirs, 
                                                             ensemble_weights_path,
                                                             exp_name)
            self.weights_file = weights_file
            if self.voting_strategy == 'weighted_model':
                self.model_weights = weights
            elif self.voting_strategy == 'weighted_families':
                self.family_weights = weights

    def _load_model(self, model_dir, config):
        """Load a single model from disk."""
        weights_path = os.path.join(model_dir, 'weights.pk')
        print("loading weights from", model_dir)
        model = BaseModel(len(self.categories), window_len=config['window_len'], device=self.device)
        model.load_state_dict(tr.load(weights_path, map_location=self.device))
        model.eval()
        return model

    def fit(self, sequences_path, debug=False):
        if self.voting_strategy in ['weighted_model', 'weighted_families']:
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
                    is_training=False,
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

            stacked_preds = tr.stack(all_preds).cuda()

        if self.voting_strategy == 'weighted_model':
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

        stacked_preds = tr.stack(all_preds)
        preds, preds_bin = self._combine_ensemble_predictions(stacked_preds)
        return preds, preds_bin

    def pred_sliding(self, emb, step=4, use_softmax=False):
        all_preds = []
        all_centers = []

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

        stacked_preds = tr.stack(all_preds) 
        preds, preds_bin = self._combine_ensemble_predictions(stacked_preds)
        return centers, preds.cpu().detach()

    def _initialize_weights(self, model_dirs, ensemble_weights_path, exp_name=None):
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

        if self.voting_strategy == 'weighted_model':
            if ensemble_weights_path and os.path.exists(weights_file):
                weights = nn.Parameter(tr.load(weights_file).to(self.device))
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
        if self.voting_strategy == 'score_voting':
            pred = tr.mean(stacked_preds, dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'weighted_model':
            pred = tr.sum(stacked_preds * self.model_weights.view(-1, 1, 1), dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'weighted_families':
            pred = tr.sum(stacked_preds * self.family_weights.view(len(self.model_dirs), 1, len(self.categories)), dim=0)
            pred_bin = tr.argmax(pred, dim=1)

        elif self.voting_strategy == 'simple_voting':
            pred_classes = tr.mode(tr.argmax(stacked_preds, dim=2), dim=0)[0]
            pred = tr.nn.functional.one_hot(pred_classes, num_classes=len(self.categories)).float()
            pred_bin = tr.argmax(pred, dim=1)

        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        return pred, pred_bin
