"""
This script is adapted from the starter code published by the RF Challenge organizers at
https://github.com/RFChallenge/icassp2024rfchallenge
"""

import numpy as np
import tensorflow as tf


def compute_power(sig: np.ndarray):
    return np.mean(np.abs(sig) ** 2)


def compute_sinr(sig: np.ndarray, intrf: np.ndarray, units='dB'):
    sinr = compute_power(sig)/compute_power(intrf)
    if units == 'dB':
        return 10 * np.log10(sinr)
    return sinr


def split_to_complex(torch_tensor):
    # Splitting real and imaginary parts
    tf_tensor = tf.convert_to_tensor(torch_tensor.squeeze(0))
    real_part = tf_tensor[0, :]
    imaginary_part = tf_tensor[1, :]

    # Create complex tensor
    complex_tensor = tf.complex(real_part, imaginary_part)

    return complex_tensor


def split_to_complex_numpy(torch_tensor):
    # Splitting real and imaginary parts
    arr = torch_tensor.detach().cpu().numpy()
    real_part = arr[0, :]
    imaginary_part = arr[1, :]

    # Create complex tensor
    complex_arr = real_part + 1j * imaginary_part

    return complex_arr
