import pickle
import numpy as np
import pandas as pd
import torch as tr
from torch.utils.data import Dataset
from Bio import SeqIO

class PFamDataset(Dataset):
    """
    Sample regions of proteins with multiple family tags.
    Proteins have precomputed per-residue embeddings.
    """
    def __init__(self, dataset_path, emb_path, categories, win_len,
                 debug=True, is_training=False, sequences=None):
        """
        Initialize the PFamDataset.
        Args:
            dataset_path (str): Path to the dataset CSV file.
            emb_path (str): Path to the directory containing precomputed embeddings.
            categories (list): List of protein family categories.
            win_len (int): Length of the window to sample from the embeddings.
            debug (bool): If True, use a smaller sample for debugging.
            is_training (bool): If True, sample windows randomly; otherwise, use the center of the domain.
        """
        self.dataset = pd.read_csv(dataset_path)
        self.emb_path = emb_path
        self.categories = categories
        self.win_len = win_len
        self.is_training = is_training
        self.sequences = sequences
        if sequences is not None:
            self.sequences = {record.id: str(record.seq) for record in SeqIO.parse(sequences, "fasta")}

        if debug:
            self.dataset = self.dataset.sample(n=15)

    def __len__(self):
        return len(self.dataset)

    def soft_domain_score(self, window_start, window_end, domain_start, domain_end):
        """Compute the percentage of the interval [domain_start, domain_end] in [window_start, window_end]"""
        return max(0, (min(domain_end, window_end) - max(domain_start, window_start))/(window_end-window_start))

    def __getitem__(self, item):
        """Sample one random window from a domain entry"""
        item = self.dataset.iloc[item]

        # Load precomputed embedding or sequence
        seq = self.sequences[item.PID]
        L = len(seq)
        # Determine window center position
        if self.is_training:
            if item.end-self.win_len//2+1<=item.start+self.win_len//2:
                center = item.start+self.win_len//2 
            else:
                center = np.random.randint(item.start+self.win_len//2, item.end-self.win_len//2+1)
        else:
            center = (item.start + item.end)//2

        start = max(0, center - self.win_len//2)
        end = min(L, center + self.win_len//2)

        label = tr.zeros(len(self.categories))

        # Compute soft labels for each domain in the protein
        domains = self.dataset[self.dataset.PID==item.PID]
        for k in range(len(domains)):
            score = self.soft_domain_score(start, end, domains.iloc[k].start, domains.iloc[k].end)
            label_ind = self.categories.index(domains.iloc[k].PF)
            label[label_ind] = max(score, label[label_ind])

        # Force labels to sum 1
        s = label.sum()
        if s<1:
            ind = tr.where(label==0)[0]
            label[ind] = (1-s)/len(ind)

        # Extract window
        if self.emb_path is not None: # embedding is providded
            win = tr.zeros((emb.shape[0], self.win_len), dtype=tr.float)
            win[:,:end-start] = tr.tensor(emb[:, start:end], dtype=tr.float)
        else: # embedding is computed on-the-fly
            
            # window seq (this has very low context and do not work well)
            # win = seq[start:end] 
            
            # returning full seq (this should replicate baseline usage of embeddings)
            win = seq 
            
            # return domain sequence (not valid in test)
            #win = seq[item.start:item.end]  
            #start = max(0, start - item.start)
            #end = end - item.start
            
        return win, label, item.PID, start, end
