import os
import sys
import time
import torch as tr
import torch.multiprocessing
from torch.utils.data import DataLoader, SubsetRandomSampler
from src.dataset import PFamDataset
#from src.basemodel import BaseModel as BaseModel # TODO fix from the config
from src.basemodel_lora import BaseModelLoRA as BaseModel
torch.multiprocessing.set_sharing_strategy('file_system')


class WarmupScheduler:
    """Learning rate scheduler with warmup."""
    def __init__(self, optimizer, warmup_epochs, base_lrs):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lrs = base_lrs
        self.current_epoch = 0
        
    def step(self):
        """Call at the start of each epoch."""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # Restore base learning rates after warmup
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i]

def train(config, categories, output_folder):
    """
    Trains a base model.
    Args:
        config (dict): Configuration dictionary containing training parameters.
        categories (list): List of categories for the model.
        output_folder (str): Directory to save model weights and training summary.
    """
    # Define paths for saving model weights and training summary
    filename = os.path.join(output_folder, "weights.pk")
    summary = os.path.join(output_folder, "train_summary.csv")

    # Load training and validation datasets
    train_data = PFamDataset(f"{config['data_path']}train.csv", config["emb_path"],
                            categories, win_len=config['window_len'],
                            is_training=True, sequences=f"{config['data_path']}train.fasta", use_embeddings=config['use_embeddings'])
    dev_data = PFamDataset(f"{config['data_path']}dev.csv", config["emb_path"],
                        categories, win_len=config['window_len'],
                        is_training=False, sequences=f"{config['data_path']}dev.fasta", use_embeddings=config['use_embeddings'])
    
    # Limit validation set in debug mode
    if config.get('debug', False):
        dev_data.dataset = dev_data.dataset.head(200)
        print(f"DEBUG MODE: Validation limited to {len(dev_data)} sequences")

    # Get train_fraction parameter (default to 1.0 for backward compatibility)
    train_fraction = config.get('train_fraction', 1.0)
    use_sampling = train_fraction < 1.0

    if use_sampling:
        num_train_samples = int(len(train_data) * train_fraction)
        print(f"train {len(train_data)} (using {num_train_samples} samples per epoch, {train_fraction*100:.1f}%), dev {len(dev_data)}")
    else:
        print("train", len(train_data), "dev", len(dev_data))

    # Create validation data loader (fixed across epochs) - with optimizations
    dev_loader = DataLoader(
        dev_data, 
        batch_size=config['batch_size'],
        num_workers=config.get('nworkers', 1),
        pin_memory=True,
        persistent_workers=config.get('nworkers', 1) > 0,
        prefetch_factor=config.get('prefetch_factor', 2)
    )

    # Initialize the model
    net = BaseModel(
        len(categories),
        window_len=config['window_len'],
        lr_lora=config['lr_lora'],
        lr_cnn=config['lr_cnn'],
        lr_fc=config['lr_fc'],
        device=config['device'],
        freeze_cnn_fc=config.get('freeze_cnn_fc', False),
        use_amp=config.get('use_amp', True),
        accumulation_steps=config.get('accumulation_steps', 1)
    )
    
    # Setup warmup scheduler if configured
    warmup_epochs = config.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        base_lrs = [param_group['lr'] for param_group in net.optim.param_groups]
        warmup_scheduler = WarmupScheduler(net.optim, warmup_epochs, base_lrs)
        print(f"Using warmup for first {warmup_epochs} epochs")
    else:
        warmup_scheduler = None
    
    # Load pretrained weights if specified
    if config.get('pretrained_path'):
        net.load_cnn_fc_weights(config['pretrained_path'])
        
        # Freeze parameters if specified
        if config.get('freeze_cnn'):
            net.freeze_cnn_params()
        if config.get('freeze_fc'):
            net.freeze_fc_params()
        if config.get('freeze_cnn_fc'):
            net.freeze_cnn_fc_params()

    # Check if a previous model exists
    if os.path.exists(filename):
        if config['continue_training']:
            # Load the model and training state if continuing training
            print(f"Loading model from {filename}")
            net.load_state_dict(tr.load(filename))
            with open(summary, 'r') as s:
                last_sum = s.readlines()[-1].split(',')
                INIT_EP = int(last_sum[0]) + 1
                best_err = float(last_sum[4])
                counter = int(last_sum[5])
        else:
            # Prompt the user to confirm overwriting the existing model
            print(f"Previous model found in {filename} and 'continue_training' is set to False.")
            confirmation = input("Are you sure you want to overwrite the existing model? (yes/[no]): ")
            if confirmation.lower() == "yes":
                os.remove(filename)
                os.remove(summary)
                print("Previous model deleted successfully. Starting training...")
                with open(summary, 'w') as s:
                    s.write("Ep,Train loss,Dev Loss,Dev error,Best error,Counter,Epoch time (s)\n")
                    INIT_EP, counter, best_err = 0, 0, 999.0
            else:
                print("Deletion aborted. Set 'continue_training' parameter to 'True' in config.json if you want to continue training an already existing model.")
                sys.exit()
    else:
        # Initialize training state if no previous model exists
        with open(summary, 'w') as s:
            s.write("Ep,Train loss,Dev Loss,Dev error,Best error,Counter,Epoch time (s)\n")
            INIT_EP, counter, best_err = 0, 0, 999.0

    # Training loop
    dev_loss, dev_err, *_ = net.pred(dev_loader)
    print(f"Initial dev loss {dev_loss:.3f}, dev err {dev_err:.3f}")
    for epoch in range(INIT_EP, config['nepoch']):
        start_time = time.time()

        # Apply warmup if configured
        if warmup_scheduler is not None:
            warmup_scheduler.step()
            current_lrs = [f"{pg['lr']:.2e}" for pg in net.optim.param_groups]
            print(f"  LR: {current_lrs}")
        
        # Create train loader (with sampling if enabled) - with optimizations
        if use_sampling:
            # Generate random indices for this epoch
            indices = tr.randperm(len(train_data))[:num_train_samples].tolist()
            sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(
                train_data, 
                batch_size=config['batch_size'],
                sampler=sampler, 
                num_workers=config.get('nworkers', 1),
                pin_memory=True,
                persistent_workers=config.get('nworkers', 1) > 0,
                prefetch_factor=config.get('prefetch_factor', 2)
            )
        else:
            # Create loader once if not using sampling
            if epoch == INIT_EP:
                train_loader = DataLoader(
                    train_data, 
                    batch_size=config['batch_size'],
                    shuffle=True, 
                    num_workers=config.get('nworkers', 1),
                    pin_memory=True,
                    persistent_workers=config.get('nworkers', 1) > 0,
                    prefetch_factor=config.get('prefetch_factor', 2)
                )

        train_loss = net.fit(train_loader)
        dev_loss, dev_err, *_ = net.pred(dev_loader)

        # Early stopping
        sv_mod = ""
        if dev_err < best_err:
            # Save the model if validation error improves
            best_err = dev_err
            tr.save(net.state_dict(), filename)
            counter = 0
            sv_mod = " - MODEL SAVED"
        else:
            counter += 1
            sv_mod = f" - EPOCH {counter} of {config['patience']}"

        epoch_time = time.time() - start_time

        print_msg = f"{epoch}: train loss {train_loss:.3f}, dev loss {dev_loss:.3f}, dev err {dev_err:.3f}"
        print(print_msg + sv_mod + f" - Time: {epoch_time:.2f}s")

        with open(summary, 'a') as s:
            s.write(f"{epoch},{train_loss},{dev_loss},{dev_err},{best_err},{counter},{epoch_time:.2f}\n")

        # Check for early stopping condition
        if counter >= config['patience']:
            break
