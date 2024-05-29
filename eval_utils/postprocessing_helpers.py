#!/usr/bin/python3
"""
module for postprocessing helpers
"""

from scipy.stats import binned_statistic
from comm_utils import qpsk_helper, ofdm_helper
from . import plot_competition_figures
import numpy as np
from tqdm import tqdm
import torch
import os
import h5py
from datetime import datetime

import matplotlib.pyplot as plt

# CONSTANTS
intrf_signal_set = ["CommSignal2",
                    "CommSignal3", "CommSignal5G1", "EMISignal1"]


def get_db(p): return 10*np.log10(p)
def get_pow(s): return np.mean(np.abs(s)**2, axis=-1)
def get_sinr(s, i): return get_pow(s)/get_pow(i)
def get_sinr_db(s, i): return get_db(get_sinr(s, i))


def get_mse(y, yhat): return np.mean(np.abs(y - yhat)**2, axis=-1)
def get_ber(b, bhat): return np.mean(b != bhat, axis=-1).astype(np.float32)


def evaluation_and_results(model, dataloader, soi_type, device, model_name):
    results = {
        key: {
            "sinr_all": [],
            "soi_target_all": [],
            "soi_est_all": [],
            "msg_bits_all": [],
            "soi_no_mitigation_all": [],
        } for key in intrf_signal_set
    }

    for i, (soi_mix, soi_target, msg_bits, intrf_type, sinr) in enumerate(tqdm(dataloader, desc="Evaluation", unit="batch")):
        soi_mix = soi_mix.to(device)
        with torch.no_grad():
            soi_est = model(soi_mix)
        for idx, int_type in enumerate(intrf_signal_set):
            results[int_type]["sinr_all"].extend(
                sinr[intrf_type == idx].numpy())
            results[int_type]["soi_target_all"].extend(
                soi_target[intrf_type == idx].numpy())
            results[int_type]["soi_est_all"].extend(
                soi_est[intrf_type == idx].cpu().numpy())
            results[int_type]["msg_bits_all"].extend(
                msg_bits[intrf_type == idx].numpy())
            results[int_type]["soi_no_mitigation_all"].extend(
                soi_mix[intrf_type == idx].cpu().numpy())

    # combine IQ channels
    for intrf_type in intrf_signal_set:
        results[intrf_type]["sinr_all"] = np.array(
            results[intrf_type]["sinr_all"])
        results[intrf_type]["soi_target_all"] = np.array(
            results[intrf_type]["soi_target_all"])[:, 0, :] + 1j * np.array(results[intrf_type]["soi_target_all"])[:, 1, :]
        results[intrf_type]["soi_est_all"] = np.array(
            results[intrf_type]["soi_est_all"])[:, 0, :] + 1j * np.array(results[intrf_type]["soi_est_all"])[:, 1, :]
        results[intrf_type]["msg_bits_all"] = np.array(
            results[intrf_type]["msg_bits_all"])
        results[intrf_type]["soi_no_mitigation_all"] = np.array(
            results[intrf_type]["soi_no_mitigation_all"])[:, 0, :] + 1j * np.array(results[intrf_type]["soi_no_mitigation_all"])[:, 1, :]

    # evaluate the mse and ber
    eval_metrics = {
        key: {
            "sinr_all": [],
            "mse_no_mitigation": [],
            "mse_est": [],
            "ber_no_mitigation": [],
            "ber_est": []
        } for key in intrf_signal_set
    }

    for intrf_type in intrf_signal_set:
        y = results[intrf_type]["soi_target_all"]
        yhat = results[intrf_type]["soi_no_mitigation_all"]
        eval_metrics[intrf_type]["mse_no_mitigation"] = get_mse(y, yhat)
        yhat = results[intrf_type]["soi_est_all"]
        eval_metrics[intrf_type]["mse_est"] = get_mse(y, yhat)

        # demodulate the signals
        b = results[intrf_type]["msg_bits_all"]
        if soi_type == "QPSK":
            bhat_no_mitigation, _ = qpsk_helper.demodulate_qpsk_signal(
                results[intrf_type]["soi_no_mitigation_all"])
            bhat, _ = qpsk_helper.demodulate_qpsk_signal(
                results[intrf_type]["soi_est_all"])
        elif soi_type == "QPSK_OFDM":
            resource_grid = ofdm_helper.create_resource_grid(40960//80)
            bhat_no_mitigation, _ = ofdm_helper.demodulate_ofdm_signal(
                results[intrf_type]["soi_no_mitigation_all"], resource_grid)
            bhat, _ = ofdm_helper.demodulate_ofdm_signal(
                results[intrf_type]["soi_est_all"], resource_grid)
        eval_metrics[intrf_type]["ber_no_mitigation"] = get_ber(
            b, bhat_no_mitigation)
        eval_metrics[intrf_type]["ber_est"] = get_ber(b, bhat)

        # save the sinr
        eval_metrics[intrf_type]["sinr_all"] = results[intrf_type]["sinr_all"]

    os.makedirs("figures", exist_ok=True)
    fig_mse, axes_mse = plt.subplots(2, 2, figsize=(10, 8))
    fig_ber, axes_ber = plt.subplots(2, 2, figsize=(10, 8))

    rows = [0, 0, 1, 1]
    cols = [0, 1, 0, 1]

    for i, intrf_type in enumerate(intrf_signal_set):
        # plot MSE
        sinr_all = eval_metrics[intrf_type]["sinr_all"]
        mse_no_mitigation = eval_metrics[intrf_type]["mse_no_mitigation"]
        axes_mse[rows[i], cols[i]].semilogy(
            *get_smoothed(sinr_all, mse_no_mitigation, 11), "o--", linewidth=3, label="No Mitigation")
        mse_est = eval_metrics[intrf_type]["mse_est"]
        axes_mse[rows[i], cols[i]].semilogy(
            *get_smoothed(sinr_all, mse_est, 11), "o--", linewidth=3, label=model_name)
        axes_mse[rows[i], cols[i]].set_title(
            f"{soi_type}_{intrf_type}", fontsize=14)
        axes_mse[rows[i], cols[i]].grid()
        axes_mse[rows[i], cols[i]].set_xlabel('SINR (dB)', fontsize=12)
        if cols[i] == 0:
            axes_mse[rows[i], cols[i]].set_ylabel(
                'Mean Squared Error', fontsize=12)
        axes_mse[rows[i], cols[i]].tick_params(axis='x', labelsize=10)
        axes_mse[rows[i], cols[i]].tick_params(axis='y', labelsize=10)

        # plot BER
        ber_no_mitigation = eval_metrics[intrf_type]["ber_no_mitigation"]
        axes_ber[rows[i], cols[i]].semilogy(
            *get_smoothed(sinr_all, ber_no_mitigation, 11), "o--", linewidth=3, label="No Mitigation")
        ber_est = eval_metrics[intrf_type]["ber_est"]
        axes_ber[rows[i], cols[i]].semilogy(
            *get_smoothed(sinr_all, ber_est, 11), "o--", linewidth=3, label=model_name)
        axes_ber[rows[i], cols[i]].plot(
            *get_smoothed(sinr_all, 10**-2 * np.ones(sinr_all.shape[0]), 11), "o", color="grey", markersize=2)
        axes_ber[rows[i], cols[i]].set_title(
            f"{soi_type}_{intrf_type}", fontsize=14)
        axes_ber[rows[i], cols[i]].grid()
        axes_ber[rows[i], cols[i]].set_xlabel('SINR (dB)', fontsize=12)
        if cols[i] == 0:
            axes_ber[rows[i], cols[i]].set_ylabel(
                'Bit Error Rate', fontsize=12)
        axes_ber[rows[i], cols[i]].tick_params(axis='x', labelsize=10)
        axes_ber[rows[i], cols[i]].tick_params(axis='y', labelsize=10)

    fig_mse.subplots_adjust(hspace=0.75)
    fig_mse.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2,
                   labels=["No mitigation", model_name], fontsize=14)
    fig_ber.subplots_adjust(hspace=0.75)
    fig_ber.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=2,
                   labels=["No mitigation", model_name], fontsize=14)
    # Save the figures
    mse_path = f'figures/{soi_type}_{model_name}_competition_mse.png'
    fig_mse.savefig(mse_path, dpi=800, bbox_inches='tight')
    print("MSE figure saved at {}".format(mse_path))
    ber_path = f'figures/{soi_type}_{model_name}_competition_ber.png'
    fig_ber.savefig(ber_path, dpi=800, bbox_inches='tight')
    print("BER figure saved at {}".format(ber_path))


def evaluation_and_results2(models, dataloaders, soi_type, interference_type, device, model_names, standardized):
    if interference_type == 'all':
        intrf_signal_set2 = intrf_signal_set
    else:
        intrf_signal_set2 = [interference_type]

    model_names = ["No Mitigation"] + list(model_names)
    models = [None] + list(models)
    standardized = [False] + list(standardized)

    results = {
        key: {
            key2: {
                'sinr_all': [],
                'soi_target_all': [],
                'soi_est_all': [],
                'msg_bits_all': [],
            } for key2 in intrf_signal_set2
        } for key in model_names
    }

    for i, ((soi_mix_r, soi_target_r, msg_bits_r, intrf_type_r, sinr_r), (
            soi_mix, soi_target, msg_bits, intrf_type, sinr)) in enumerate(tqdm(zip(dataloaders[0], dataloaders[1]), desc="Evaluation", unit="batch")):
        soi_mix = soi_mix.to(device)
        soi_mix_r = soi_mix_r.to(device)
        for model_name, model, std in zip(model_names, models, standardized):
            if model_name == "No Mitigation":
                soi_est = soi_mix_r
            else:
                model.eval()
                with torch.no_grad():
                    soi_est = model(soi_mix_r) if not std else model(soi_mix)
            if interference_type == "all":
                for idx, int_type in enumerate(intrf_signal_set2):
                    results[model_name][int_type]["sinr_all"].extend(
                        sinr_r[intrf_type == idx].numpy() if not std else sinr[intrf_type == idx].numpy())
                    results[model_name][int_type]["soi_target_all"].extend(
                        soi_target_r[intrf_type == idx].numpy() if not std else soi_target[intrf_type == idx].numpy())
                    results[model_name][int_type]["soi_est_all"].extend(
                        soi_est[intrf_type == idx].cpu().numpy())
                    results[model_name][int_type]["msg_bits_all"].extend(
                        msg_bits_r[intrf_type == idx].numpy() if not std else msg_bits[intrf_type == idx].numpy())
            else:
                idx = intrf_signal_set.index(interference_type)
                results[model_name][interference_type]["sinr_all"].extend(
                    sinr_r[intrf_type == idx].numpy() if not std else sinr[intrf_type == idx].numpy())
                results[model_name][interference_type]["soi_target_all"].extend(
                    soi_target_r[intrf_type == idx].numpy() if not std else soi_target[intrf_type == idx].numpy())
                results[model_name][interference_type]["soi_est_all"].extend(
                    soi_est[intrf_type == idx].cpu().numpy())
                results[model_name][interference_type]["msg_bits_all"].extend(
                    msg_bits_r[intrf_type == idx].numpy() if not std else msg_bits[intrf_type == idx].numpy())

    for intrf_type in intrf_signal_set2:
        for model_name in model_names:
            results[model_name][intrf_type]["sinr_all"] = np.array(
                results[model_name][intrf_type]["sinr_all"])
            results[model_name][intrf_type]["soi_target_all"] = np.array(
                results[model_name][intrf_type]["soi_target_all"])[:, 0, :] + 1j * np.array(results[model_name][intrf_type]["soi_target_all"])[:, 1, :]  # combine IQ channels
            results[model_name][intrf_type]["soi_est_all"] = np.array(
                results[model_name][intrf_type]["soi_est_all"])[:, 0, :] + 1j * np.array(results[model_name][intrf_type]["soi_est_all"])[:, 1, :]  # combine IQ channels
            results[model_name][intrf_type]["msg_bits_all"] = np.array(
                results[model_name][intrf_type]["msg_bits_all"])

    # evaluate the mse and ber
    eval_metrics = {
        key: {
            key2: {
                'sinr_all': [],
                'mse_est': [],
                'ber_est': []
            } for key2 in intrf_signal_set2
        } for key in model_names
    }

    for intrf_type in intrf_signal_set2:
        for model_name in model_names:
            y = results[model_name][intrf_type]["soi_target_all"]
            yhat = results[model_name][intrf_type]["soi_est_all"]
            eval_metrics[model_name][intrf_type]["mse_est"] = get_mse(y, yhat)

            # demodulate the signals
            b = results[model_name][intrf_type]["msg_bits_all"]
            if soi_type == "QPSK":
                bhat, _ = qpsk_helper.demodulate_qpsk_signal(
                    results[model_name][intrf_type]["soi_est_all"])
            elif soi_type == "QPSK_OFDM":
                resource_grid = ofdm_helper.create_resource_grid(40960//80)
                bhat, _ = ofdm_helper.demodulate_ofdm_signal(
                    results[model_name][intrf_type]["soi_est_all"], resource_grid)
            eval_metrics[model_name][intrf_type]["ber_est"] = get_ber(b, bhat)

            # save the sinr
            eval_metrics[model_name][intrf_type]["sinr_all"] = results[model_name][intrf_type]["sinr_all"]

    output_directory = f'outputs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_directory, exist_ok=True)
    # save results to file
    output_file = f"{output_directory}/{soi_type}_{interference_type}_eval_metrics.h5"
    with h5py.File(output_file, "w") as f:
        for key, subdict in eval_metrics.items():
            for key2, value in subdict.items():
                for key3, value2 in value.items():
                    f.create_dataset(f"{key}_{key2}_{key3}", data=value2)
    print("\n\nResults saved at {}".format(output_file))

    figure_directory = f'figures_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(figure_directory, exist_ok=True)
    nmodels = len(model_names)
    if interference_type == 'all':
        fig_mse, axes_mse = plt.subplots(2, 2, figsize=(10, 8))
        fig_ber, axes_ber = plt.subplots(2, 2, figsize=(10, 8))
        rows = [0, 0, 1, 1]
        cols = [0, 1, 0, 1]
        for i, intrf_type in enumerate(intrf_signal_set2):
            for model_name in model_names:
                # plot MSE
                sinr_all = eval_metrics[model_name][intrf_type]["sinr_all"]
                mse_est = eval_metrics[model_name][intrf_type]["mse_est"]
                axes_mse[rows[i], cols[i]].semilogy(
                    *get_smoothed(sinr_all, mse_est, 11), "o--", linewidth=3, label=model_name)
                axes_mse[rows[i], cols[i]].set_title(
                    f"{soi_type}_{intrf_type}", fontsize=14)
                axes_mse[rows[i], cols[i]].grid()
                axes_mse[rows[i], cols[i]].set_xlabel('SINR (dB)', fontsize=12)
                if cols[i] == 0:
                    axes_mse[rows[i], cols[i]].set_ylabel(
                        'Mean Squared Error', fontsize=12)
                axes_mse[rows[i], cols[i]].tick_params(axis='x', labelsize=10)
                axes_mse[rows[i], cols[i]].tick_params(axis='y', labelsize=10)

                # plot BER
                ber_est = eval_metrics[model_name][intrf_type]["ber_est"]
                axes_ber[rows[i], cols[i]].semilogy(
                    *get_smoothed(sinr_all, ber_est, 11), "o--", linewidth=3, label=model_name)
                axes_ber[rows[i], cols[i]].set_title(
                    f"{soi_type}_{intrf_type}", fontsize=14)
                axes_ber[rows[i], cols[i]].grid()
                axes_ber[rows[i], cols[i]].set_xlabel('SINR (dB)', fontsize=12)
                if cols[i] == 0:
                    axes_ber[rows[i], cols[i]].set_ylabel(
                        'Bit Error Rate', fontsize=12)
                axes_ber[rows[i], cols[i]].tick_params(axis='x', labelsize=10)
                axes_ber[rows[i], cols[i]].tick_params(axis='y', labelsize=10)

        fig_mse.subplots_adjust(hspace=0.75)
        fig_mse.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=nmodels,
                       labels=model_names, fontsize=14)
        fig_ber.subplots_adjust(hspace=0.75)
        fig_ber.legend(loc='upper center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=nmodels,
                       labels=model_names, fontsize=14)
        # Save the figures
        mse_path = f'{figure_directory}/{soi_type}_{interference_type}_competition_mse.png'
        fig_mse.savefig(mse_path, dpi=800, bbox_inches='tight')
        print("MSE figure saved at {}".format(mse_path))
        ber_path = f'{figure_directory}/{soi_type}_{interference_type}_competition_ber.png'
        fig_ber.savefig(ber_path, dpi=800, bbox_inches='tight')
        print("BER figure saved at {}".format(ber_path))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for model_name in model_names:
            # plot MSE
            sinr_all = eval_metrics[model_name][interference_type]["sinr_all"]
            mse_est = eval_metrics[model_name][interference_type]["mse_est"]
            axes[0].semilogy(
                *get_smoothed(sinr_all, mse_est, 11), "o--", linewidth=3, label=model_name)
            axes[0].set_title(
                f"{soi_type}_{interference_type}", fontsize=14)
            axes[0].grid()
            axes[0].set_xlabel('SINR (dB)', fontsize=12)
            axes[0].set_ylabel('Mean Squared Error', fontsize=12)
            axes[0].tick_params(axis='x', labelsize=10)
            axes[0].tick_params(axis='y', labelsize=10)

            # plot BER
            ber_est = eval_metrics[model_name][interference_type]["ber_est"]
            axes[1].semilogy(
                *get_smoothed(sinr_all, ber_est, 11), "o--", linewidth=3, label=model_name)
            axes[1].plot(*get_smoothed(sinr_all, 10**-2 * np.ones(sinr_all.shape[0]), 11),
                         "o", color="grey", markersize=2, label=None)
            axes[1].set_title(
                f"{soi_type}_{interference_type}", fontsize=14)
            axes[1].grid()
            axes[1].set_xlabel('SINR (dB)', fontsize=12)
            axes[1].set_ylabel('Bit Error Rate', fontsize=12)
            axes[1].tick_params(axis='x', labelsize=10)
            axes[1].tick_params(axis='y', labelsize=10)

        fig.subplots_adjust(bottom=0.3)
        fig.legend(loc='upper center', labels=model_names, bbox_to_anchor=(
            0.5, 1.1), ncols=nmodels, fancybox=True, shadow=True, fontsize=10)
        fig.savefig(f"{figure_directory}/{soi_type}_{interference_type}_competition.png",
                    dpi=800, bbox_inches='tight')
        print(
            f"Figure saved at {figure_directory}/{soi_type}_{interference_type}_competition.png")


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


def get_smoothed(sinr: np.ndarray, ber, npoints=50):
    bins = np.linspace(sinr.min(), sinr.max(), npoints)
    bin_means, bin_edges, _ = binned_statistic(
        sinr, ber, statistic='mean', bins=bins)

    # To plot the line at the center of the bins, calculate the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, bin_means


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
