"""
This script is adapted from the starter code published by the RF Challenge organizers at
https://github.com/RFChallenge/icassp2024rfchallenge
"""

import numpy as np


def compute_power(sig: np.ndarray):
    return np.mean(np.abs(sig) ** 2)


def compute_sinr(sig: np.ndarray, intrf: np.ndarray, units='dB'):
    sinr = compute_power(sig)/compute_power(intrf)
    if units == 'dB':
        return 10 * np.log10(sinr)
    return sinr
