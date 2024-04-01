from comm_utils import demodulate_qpsk_signal, demodulate_ofdm_signal
import torch
from torch.nn import MSELoss
from tqdm import tqdm
import numpy as np


def evaluate_competition(model, dataset, soi_type, device,
                         intrf_sig_names=('CommSignal2', 'CommSignal3', 'CommSignal5G1', 'EMISignal1')):
    if soi_type == 'QPSK':
        demodulator = demodulate_qpsk_signal
    elif soi_type == 'QPSK_OFDM':
        demodulator = demodulate_ofdm_signal
    else:
        raise NotImplementedError

    model.eval()
    mse_loss_model = np.zeros((len(intrf_sig_names),))
    ber_model = np.zeros((len(intrf_sig_names),))
    mse_loss = np.zeros((len(intrf_sig_names),))
    ber = np.zeros((len(intrf_sig_names),))

    with torch.no_grad():
        with tqdm(dataset, desc='Evaluation', unit='Sample') as pbar:
            for i, sample in enumerate(pbar):
                sig_mixed, sig_target, msg_bits, intrf_label, sinr_db = sample
                sig_pred = model(sig_mixed)
                mse_loss_model[intrf_label] += MSELoss(sig_pred, sig_target).item()
                mse_loss[intrf_label] += MSELoss(sig_mixed, sig_target).item()

                bits, _ = demodulator(sig_mixed)
                bits_model, _ = demodulator(sig_pred)
                ber[intrf_label] += np.sum(msg_bits != bits) / len(msg_bits)
                ber_model[intrf_label] += np.sum(msg_bits != bits_model) / len(msg_bits)

    mse_loss_model /= (len(dataset) / len(intrf_sig_names))
    mse_loss /= (len(dataset) / len(intrf_sig_names))
    ber_model /= (len(dataset) / len(intrf_sig_names))
    ber /= (len(dataset) / len(intrf_sig_names))

    return mse_loss_model, mse_loss, ber_model, ber
