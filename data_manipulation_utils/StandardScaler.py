from joblib import load
import numpy as np


class StandardScaler:
    def __init__(self, filepaths_list):
        self.filepaths_list = filepaths_list
        self.mu = 0
        self.sigma = 1

    def fit(self):
        for filepath in self.filepaths_list:
            sig_mixed = load(filepath)[0]
            self.mu += np.mean(sig_mixed)
            self.sigma += np.mean(np.std(sig_mixed, axis=1))
        self.sigma -= 1
        self.mu /= len(self.filepaths_list)
        self.sigma /= len(self.filepaths_list)

    def __call__(self, sig_mixed):
        return (sig_mixed - self.mu) / self.sigma
