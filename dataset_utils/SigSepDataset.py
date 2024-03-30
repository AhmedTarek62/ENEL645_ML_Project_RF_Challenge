from torch.utils.data import Dataset
import os
from joblib import load


class SigSepDataset(Dataset):
    def __init__(self, dataset_dir, samples_per_batch):
        self.all_batches = [load(os.path.join(dataset_dir, batch_file)) for batch_file in os.listdir(dataset_dir)]
        self.samples_per_batch = samples_per_batch

    def __getitem__(self, index):
        file_index = index // self.samples_per_batch
        sample_index = index % self.samples_per_batch
        sig_mixed, sig_target, intrf_labels, sinr_db = self.all_batches[file_index]
        
        return sig_mixed[sample_index], sig_target[sample_index], intrf_labels[sample_index], sinr_db[sample_index]

    def __len__(self):
        return len(self.all_batches) * self.samples_per_batch

