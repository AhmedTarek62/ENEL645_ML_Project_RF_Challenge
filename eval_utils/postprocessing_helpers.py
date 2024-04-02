#!/usr/bin/python3
"""
module for postprocessing helpers
"""

from scipy.stats import binned_statistic
from comm_utils import qpsk_helper, ofdm_helper
import numpy as np

import matplotlib.pyplot as plt


def get_db(p): return 10*np.log10(p)
def get_pow(s): return np.mean(np.abs(s)**2, axis=-1)
def get_sinr(s, i): return get_pow(s)/get_pow(i)


def get_sinr_db(s, i): return get_db(get_sinr(s, i))


def get_smoothed(sinr: np.ndarray, ber, npoints=50):
    bins = np.linspace(sinr.min(), sinr.max(), npoints)
    bin_means, bin_edges, _ = binned_statistic(
        sinr, ber, statistic='mean', bins=bins)

    # To plot the line at the center of the bins, calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, bin_means


def postprocess_qpsk(sig_est, sig_no_mitigation, sig_target, msg_bits, sinr_db):
    """
    Post process the QPSK signals and calculate the MSE and BER
    """
    # calculate the mean square error
    mse_no_mitigation = get_db(
        np.mean(np.abs(sig_no_mitigation - sig_target)**2, axis=-1))
    mse_est = get_db(np.mean(np.abs(sig_est - sig_target)**2, axis=-1))

    # calculate the bit error rate
    bit_no_mitigation, _ = qpsk_helper.demodulate_qpsk_signal(
        sig_no_mitigation)
    bit_est, _ = qpsk_helper.demodulate_qpsk_signal(sig_est)

    ber_no_mitigation = np.mean(
        bit_no_mitigation != msg_bits, axis=-1).astype(np.float32)
    ber_est = np.mean(bit_est != msg_bits, axis=-1).astype(np.float32)

    return {
        "mse_no_mitigation": mse_no_mitigation,
        "mse_est": mse_est,
        "ber_no_mitigation": ber_no_mitigation,
        "ber_est": ber_est,
        "sinr_db": sinr_db
    }


def postprocess_ofdm(sig_est, sig_no_mitigation, sig_target, msg_bits, sinr_db):
    """
    Post process the OFDM signals and calculate the MSE and BER
    """
    # calculate the mean square error
    mse_no_mitigation = get_db(
        np.mean(np.abs(sig_no_mitigation - sig_target)**2, axis=-1))
    mse_est = get_db(np.mean(np.abs(sig_est - sig_target)**2, axis=-1))

    # calculate the bit error rate
    bit_no_mitigation, _ = ofdm_helper.demodulate_ofdm_signal(
        sig_no_mitigation)
    bit_est, _ = ofdm_helper.demodulate_ofdm_signal(sig_est)

    ber_no_mitigation = np.mean(
        bit_no_mitigation != msg_bits, axis=-1).astype(np.float32)
    ber_est = np.mean(bit_est != msg_bits, axis=-1).astype(np.float32)

    return {
        "mse_no_mitigation": mse_no_mitigation,
        "mse_est": mse_est,
        "ber_no_mitigation": ber_no_mitigation,
        "ber_est": ber_est,
        "sinr_db": sinr_db
    }


def visualize_results(results, soi_type, interference_type, model_name="Default_Torch_WaveNet", smoothen=True):
    """
    Visualize the results
    """
    sinr_all = results["sinr_db"]
    mse_no_mitigation = results["mse_no_mitigation"]
    mse_est = results["mse_est"]
    ber_no_mitigation = results["ber_no_mitigation"]
    ber_est = results["ber_est"]

    if smoothen:
        sinr_db, mse_no_mitigation = get_smoothed(
            sinr_all, mse_no_mitigation, 11)
        sinr_db, mse_est = get_smoothed(sinr_all, mse_est, 11)
        sinr_db, ber_no_mitigation = get_smoothed(
            sinr_all, ber_no_mitigation, 11)
        sinr_db, ber_est = get_smoothed(sinr_all, ber_est, 11)
    else:
        sinr_db, mse_no_mitigation = zip(
            *sorted(dict(zip(sinr_all, mse_no_mitigation)).items()))
        sinr_db, mse_est = zip(*sorted(dict(zip(sinr_all, mse_est)).items()))
        sinr_db, ber_no_mitigation = zip(
            *sorted(dict(zip(sinr_all, ber_no_mitigation)).items()))
        sinr_db, ber_est = zip(*sorted(dict(zip(sinr_all, ber_est)).items()))

    fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    axes[0].plot(sinr_db, mse_no_mitigation, "x--",
                 label="Default_NoMitigation")
    axes[0].plot(sinr_db, mse_est, "x--", label=model_name)
    axes[0].grid()
    axes[0].legend()
    axes[0].set_title(f"MSE - {soi_type} + {interference_type}")
    axes[0].set_xlabel("SINR [dB]")
    axes[0].set_ylabel("MSE")

    axes[1].semilogy(
        sinr_db, ber_no_mitigation, "x--", label="Default_NoMitigation")
    axes[1].semilogy(
        sinr_db, ber_est, "x--", label=model_name)

    axes[1].grid()
    axes[1].legend()
    axes[1].set_title(f"BER - {soi_type} + {interference_type}")
    axes[1].set_xlabel("SINR [dB]")
    axes[1].set_ylabel("BER")

    fig.savefig(f"results_{soi_type}_{interference_type}.png")
