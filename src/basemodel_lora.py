import math
import torch as tr
from torch import nn
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from peft import LoraConfig, get_peft_model

class BaseModelLoRA(nn.Module): 
    """
    ESM2 (with or without LoRA) + convolutional neural network with residual layers for protein family classification.
    """
    def __init__(self, nclasses, lr_lora, lr_cnn, lr_fc, emb_size=1280,  device="cuda", 
                 logger=None, filters=1100, kernel_size=9, num_layers=5, 
                 first_dilated_layer=2, dilation_rate=3, resnet_bottleneck_factor=.5, use_lora=True,
                 freeze_cnn_fc=False):
        super().__init__()

        self.use_lora = use_lora
        self.freeze_cnn_fc = freeze_cnn_fc
        self.emb_model, alphabet = tr.hub.load("facebookresearch/esm:main",
                              "esm2_t33_650M_UR50D")
        self.batch_converter = alphabet.get_batch_converter()

        if use_lora:
            # finetune all layers
            target_layers = []
            for name, module in self.emb_model.named_modules():
                if isinstance(module, tr.nn.Linear):
                    target_layers.append(name)

            lora_config = LoraConfig(
                r=8,                 # low-rank
                lora_alpha=32,
                target_modules=target_layers, 
                lora_dropout=0.05,
                bias="none",
                #task_type="CAUSAL_LM"  # this value is ignored for some models, but keep sensible default
            )

            self.emb_model = get_peft_model(self.emb_model, lora_config)
            print(self.emb_model)
        else:
            # Freeze ESM2 parameters when not using LoRA
            for param in self.emb_model.parameters():
                param.requires_grad = False
            print("ESM2 loaded without LoRA (frozen)")
        
        self.emb_size = emb_size 

        self.logger = logger
        self.train_steps = 0
        self.dev_steps = 0

        self.cnn = [nn.Conv1d(self.emb_size, filters, kernel_size, padding="same")]
        for k in range(num_layers):
            self.cnn.append(ResidualLayer(k, first_dilated_layer, dilation_rate, 
                                          resnet_bottleneck_factor, filters, kernel_size))
        self.cnn.append(nn.AdaptiveMaxPool1d(1))
        self.cnn = nn.Sequential(*self.cnn)

        self.fc = nn.Linear(filters, nclasses) 

        self.loss = nn.CrossEntropyLoss()
        # Configure optimizer based on LoRA usage and frozen components
        if use_lora:
            if freeze_cnn_fc:
                # Only optimize ESM2 LoRA parameters when CNN/FC are frozen
                self.optim = tr.optim.AdamW([
                    {"params": self.emb_model.parameters(), "lr": lr_lora, "weight_decay": 0.0}
                ])
            else:
                # Include ESM2 LoRA parameters + CNN/FC in optimizer
                self.optim = tr.optim.AdamW([
                    {"params": self.emb_model.parameters(), "lr": lr_lora, "weight_decay": 0.0},  
                    {"params": self.cnn.parameters(), "lr": lr_cnn, "weight_decay": 0.01},       
                    {"params": self.fc.parameters(), "lr": lr_fc, "weight_decay": 0.01}         
                ])
        else:
            # Only optimize CNN and FC when ESM2 is frozen
            self.optim = tr.optim.AdamW([
                {"params": self.cnn.parameters(), "lr": lr_cnn, "weight_decay": 0.01},       
                {"params": self.fc.parameters(), "lr": lr_fc, "weight_decay": 0.01}         
            ])

        self.to(device)
        self.device = device

        print("BaseModelLoRA initialized with", sum(p.numel() for p in self.parameters() if p.requires_grad), "trainable parameters. ESM2 PEFT parameters : ", 
              sum(p.numel() for p in self.emb_model.parameters() if p.requires_grad))

    def load_cnn_fc_weights(self, pretrained_weights_path):
        """
        Load pretrained CNN and FC weights from a checkpoint.
        Weights are loaded but parameters remain trainable unless explicitly frozen.
        
        Args:
            pretrained_weights_path: Path to the pretrained model weights (.pk file)
        """
        print(f"Loading CNN/FC weights from {pretrained_weights_path}")
        
        # Load the pretrained state dict
        pretrained_state = tr.load(pretrained_weights_path, map_location=self.device)
        
        # Extract only CNN and FC parameters
        cnn_fc_state = {}
        for key, value in pretrained_state.items():
            if key.startswith('cnn.') or key.startswith('fc.'):
                cnn_fc_state[key] = value
        
        # Load the CNN and FC weights
        self.load_state_dict(cnn_fc_state, strict=False)
        
        print(f"Loaded {sum(p.numel() for p in self.cnn.parameters())} CNN parameters and {sum(p.numel() for p in self.fc.parameters())} FC parameters")
    
    def freeze_cnn_params(self):
        """
        Freeze CNN parameters, making them non-trainable.
        """
        for param in self.cnn.parameters():
            param.requires_grad = False
        print(f"Froze {sum(p.numel() for p in self.cnn.parameters())} CNN parameters")
    
    def freeze_fc_params(self):
        """
        Freeze FC parameters, making them non-trainable.
        """
        for param in self.fc.parameters():
            param.requires_grad = False
        print(f"Froze {sum(p.numel() for p in self.fc.parameters())} FC parameters")
    
    def freeze_cnn_fc_params(self):
        """
        Freeze both CNN and FC parameters, making them non-trainable.
        Useful for finetuning only ESM2 with LoRA.
        """
        self.freeze_cnn_params()
        self.freeze_fc_params()
        print(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        print(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def _compute_embeddings_batch(self, seq_list):
        """Compute ESM2 embeddings for a batch of sequences.
        Args:
            seq_list: List of sequences (strings)
        Returns:
            Embeddings tensor of shape [batch_size, emb_size, seq_len]
        """
        _, _, tokens = self.batch_converter([(k, s) for k, s in enumerate(seq_list)])
        
        if self.use_lora:
            # When using LoRA, compute gradients through ESM2
            emb = self.emb_model(tokens.to(self.device), repr_layers=[33])["representations"][33][:, 1:-1, :].permute(0, 2, 1)
        else:
            # When not using LoRA, freeze ESM2
            with tr.no_grad():
                emb = self.emb_model(tokens.to(self.device), repr_layers=[33])["representations"][33].permute(0, 2, 1)
        
        return emb

    def compute_embeddings(self, seq):
        """Compute ESM2 embeddings for a single sequence.
        Args:
            seq: Single sequence (string)
        Returns:
            Embeddings tensor of shape [emb_size, seq_len]
        """
        emb = self._compute_embeddings_batch([seq])
        return emb.squeeze(0)

    def forward_from_embeddings(self, emb, start, end):
        """Forward pass using pre-computed embeddings.
        Args:
            emb: Pre-computed embeddings of shape [emb_size, seq_len], [batch_size, emb_size, seq_len],
                 or [batch_size, 1, emb_size, seq_len] (when loaded from pickle files)
            start: List of start positions
            end: List of end positions
        Returns:
            Predictions tensor
        """
        # Handle different embedding shapes
        if emb.dim() == 2:
            # Shape: [emb_size, seq_len]
            emb = emb.unsqueeze(0)  # Add batch dimension -> [1, emb_size, seq_len]
        elif emb.dim() == 4:
            # Shape: [batch_size, 1, emb_size, seq_len] (from pickle files)
            # Squeeze the extra dimension
            emb = emb.squeeze(1)  # -> [batch_size, emb_size, seq_len]
        
        batch_size = emb.shape[0]
        
        emb_win = tr.zeros((batch_size, emb.shape[1], 32), dtype=tr.float).to(self.device)
        
        for k in range(batch_size):
            window_len = end[k] - start[k]
            emb_win[k, :, :window_len] = emb[k, :, start[k]:end[k]]
        
        y = self.cnn(emb_win)
        y = self.fc(y.squeeze(2))
        return y


    def forward(self, seq, start, end):
        """Forward pass computing embeddings and predictions.
        Args:
            seq: List of sequences (strings) or batch of precomputed embeddings
            start: List of start positions
            end: List of end positions
        """
        # Check if input is already embeddings (torch tensor) or sequences (list/strings)
        if isinstance(seq, tr.Tensor):
            # Input is precomputed embeddings
            emb = seq
        else:
            # Input is sequences - compute embeddings using ESM2
            emb = self._compute_embeddings_batch(seq)
        return self.forward_from_embeddings(emb, start, end)

    def fit(self, dataloader):

        avg_loss = 0
        # Set all components to training mode
        self.emb_model.train()
        self.cnn.train()
        self.fc.train()
        self.optim.zero_grad()
        for k,(x, y, _, start, end) in enumerate(tqdm(dataloader)):
            yhat = self(x, start, end)
            y = y.to(self.device)

            loss = self.loss(yhat, y)
            loss.backward()

            # Add gradient clipping to prevent exploding gradients
            tr.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            avg_loss += loss.item()
            self.optim.step()
            self.optim.zero_grad()

            if self.logger is not None:
                self.logger.add_scalar("Loss/train", loss, self.train_steps)
            self.train_steps+=1

        avg_loss /= len(dataloader)

        return avg_loss

    def pred(self, dataloader):
        test_loss = 0
        pred, ref, names, starts, ends  = [], [], [], [], []
        # Set all components to evaluation mode
        self.emb_model.eval()
        self.cnn.eval()
        self.fc.eval()
        
        for seq, y, name, start, end in tqdm(dataloader):
            with tr.no_grad():
                yhat = self(seq, start, end)
                y = y.to(self.device)
                test_loss += self.loss(yhat, y).item()

            names += name
            starts.append(start)
            ends.append(end)
            pred.append(yhat.detach().cpu())
            ref.append(y.cpu())

        pred = tr.cat(pred)
        pred_bin = tr.argmax(pred, dim=1)
        
        ref = tr.cat(ref)
        ref_bin = tr.argmax(ref, dim=1)

        self.dev_steps += 1
        test_loss /= len(dataloader)

        acc = accuracy_score(ref_bin, pred_bin)
        if self.logger is not None:
            self.logger.add_scalar("Loss/dev", test_loss, self.dev_steps)
            balacc = balanced_accuracy_score(ref_bin, pred_bin)
            self.logger.add_scalar("Error rate/dev", 1-acc, self.dev_steps)
            self.logger.add_scalar("Balanced acc/dev", balacc, self.dev_steps)

        return test_loss, 1-acc, pred, ref, names, starts, ends

class ResidualLayer(nn.Module):
    def __init__(self, layer_index, first_dilated_layer, dilation_rate, 
                 resnet_bottleneck_factor, filters, kernel_size):
        super().__init__()

        shifted_layer_index = layer_index - first_dilated_layer + 1
        dilation_rate = max(1, dilation_rate**shifted_layer_index)

        num_bottleneck_units = math.floor(
            resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(nn.BatchNorm1d(filters, track_running_stats=True),
        nn.ReLU(),
        nn.Conv1d(filters, num_bottleneck_units, kernel_size, 
                  dilation=dilation_rate, padding="same"), 
        nn.BatchNorm1d(num_bottleneck_units, track_running_stats=True),
        nn.ReLU(),
        nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"))
        # The second convolution is purely local linear transformation across
        # feature channels, as is done in
        # tensorflow_models/slim/nets/resnet_v2.bottleneck

    def forward(self, x):
        return x + self.layer(x)

