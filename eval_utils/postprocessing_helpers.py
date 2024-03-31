#!/usr/bin/python3
"""
module for postprocessing helpers
"""

from scipy.stats import binned_statistic
import numpy as np

def get_smoothed(sinr: np.ndarray, ber, npoints=50):
    bins = np.linspace(sinr.min(), sinr.max(), npoints)
    bin_means, bin_edges, _ = binned_statistic(sinr, ber, statistic='mean', bins=bins)
    
    # To plot the line at the center of the bins, calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return bin_centers, bin_means
