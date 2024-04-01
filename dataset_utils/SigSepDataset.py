from torch.utils.data import Dataset
import os
from joblib import load
import numpy as np


class SigSepDataset(Dataset):
    def __init__(self, dataset_dir, dtype='real'):
        self.filepaths_list = [os.path.join(dataset_dir, batch_file) for batch_file in os.listdir(dataset_dir)]
        self.samples_per_batch = load(self.filepaths_list[0])[0].shape[0]
        self.dtype = dtype

    @staticmethod
    def separate_real_imaginary(array):
        # Separate real and imaginary parts
        real_part = np.real(array)
        imag_part = np.imag(array)

        # Concatenate real and imaginary parts along the second dimension
        separated_array = np.concatenate((real_part, imag_part))

        return separated_array

    def __getitem__(self, index):
        file_index = index // self.samples_per_batch
        sample_index = index % self.samples_per_batch
        sig_mixed, sig_target, msg_bits, intrf_labels, sinr_db = load(self.filepaths_list[file_index])

        if self.dtype == 'real':
            return (self.separate_real_imaginary(sig_mixed[sample_index]),
                    self.separate_real_imaginary(sig_target[sample_index]),
                    msg_bits[sample_index],
                    intrf_labels[sample_index],
                    sinr_db[sample_index])

        return (sig_mixed[sample_index],
                sig_target[sample_index],
                msg_bits[sample_index],
                intrf_labels[sample_index],
                sinr_db[sample_index])

    def __len__(self):
        return len(self.filepaths_list) * self.samples_per_batch

