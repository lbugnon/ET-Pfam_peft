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
    train_data = PFamDataset(f"{config['data_path']}train.csv", config['emb_path'] if config.get('use_embeddings', True) else None,
                            categories, win_len=config['window_len'],
                            is_training=True, sequences=f"{config['data_path']}train.fasta")
    dev_data = PFamDataset(f"{config['data_path']}dev.csv", config['emb_path'] if config.get('use_embeddings', True) else None,
                        categories, win_len=config['window_len'],
                        is_training=False, sequences=f"{config['data_path']}dev.fasta")

    # Get train_fraction parameter (default to 1.0 for backward compatibility)
    train_fraction = config.get('train_fraction', 1.0)
    use_sampling = train_fraction < 1.0

    if use_sampling:
        num_train_samples = int(len(train_data) * train_fraction)
        print(f"train {len(train_data)} (using {num_train_samples} samples per epoch, {train_fraction*100:.1f}%), dev {len(dev_data)}")
    else:
        print("train", len(train_data), "dev", len(dev_data))

    # Create validation data loader (fixed across epochs)
    dev_loader = DataLoader(dev_data, batch_size=config['batch_size'],
                            num_workers=config['nworkers'])

    # Initialize the model
    net = BaseModel(len(categories), lr=config['lr'], device=config['device'])
    
    # Load pretrained weights if specified
    if config.get('pretrained_path'):
        net.load_cnn_fc_weights(config['pretrained_path'])
        
        # Freeze parameters if specified
        if config.get('freeze_cnn'):
            net.freeze_cnn_params()
        if config.get('freeze_fc'):
            net.freeze_fc_params()

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
    for epoch in range(INIT_EP, config['nepoch']):
        start_time = time.time()

        # Create train loader (with sampling if enabled)
        if use_sampling:
            # Generate random indices for this epoch
            indices = tr.randperm(len(train_data))[:num_train_samples].tolist()
            sampler = SubsetRandomSampler(indices)
            train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                                    sampler=sampler, num_workers=config['nworkers'])
        else:
            # Create loader once if not using sampling
            if epoch == INIT_EP:
                train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                                        shuffle=True, num_workers=config['nworkers'])

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
