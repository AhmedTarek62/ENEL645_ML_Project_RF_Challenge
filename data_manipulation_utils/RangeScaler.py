from joblib import load
import numpy as np


class RangeScaler:
    def __init__(self, filepaths_list, norm_range=(-1, 1)):
        self.filepaths_list = filepaths_list
        self.a, self.b = norm_range
        self.min_val_real = np.inf
        self.max_val_real = -np.inf
        self.min_val_imag = np.inf
        self.max_val_imag = -np.inf

    def fit(self):
        for filepath in self.filepaths_list:
            sig_mixed = load(filepath)[0]
            real_part = np.real(sig_mixed)
            imag_part = np.imag(sig_mixed)
            self.min_val_real = min(self.min_val_real, np.min(real_part))
            self.max_val_real = max(self.max_val_real, np.max(real_part))
            self.min_val_imag = min(self.min_val_imag, np.min(imag_part))
            self.max_val_imag = max(self.max_val_imag, np.max(imag_part))

    def __call__(self, sig_mixed):
        real_part, imag_part = np.real(sig_mixed), np.imag(sig_mixed)
        normalized_real_part = (self.a + (self.b - self.a) *
                                (real_part - self.min_val_real) / (self.max_val_real - self.min_val_real))
        normalized_imag_part = (self.a + (self.b - self.a) *
                                (imag_part - self.min_val_imag) / (self.max_val_imag - self.min_val_imag))
        return normalized_real_part + 1j * normalized_imag_part
