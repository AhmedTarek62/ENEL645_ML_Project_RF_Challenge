from comm_utils import demodulate_qpsk_signal, demodulate_ofdm_signal
import torch
from torch.nn import MSELoss
from tqdm import tqdm
import numpy as np
import tensorflow as tf


def evaluate_competition(model, dataloader, soi_type, device,
                         intrf_sig_names=('CommSignal2', 'CommSignal3', 'CommSignal5G1', 'EMISignal1')):
    if soi_type == 'QPSK':
        demodulator = demodulate_qpsk_signal
    elif soi_type == 'QPSK_OFDM':
        demodulator = demodulate_ofdm_signal
    else:
        raise NotImplementedError

    model.eval()
    all_sinr_db = np.arange(-30, 1, 3)
    mse_loss_model = np.zeros((len(intrf_sig_names), len(all_sinr_db)))
    ber_model = np.zeros((len(intrf_sig_names), len(all_sinr_db)))
    mse_loss = np.zeros((len(intrf_sig_names), len(all_sinr_db)))
    ber = np.zeros((len(intrf_sig_names), len(all_sinr_db)))
    criterion = MSELoss()
    with torch.no_grad():
        with tqdm(dataloader, desc='Evaluation', unit='Sample') as pbar:
            for i, sample in enumerate(pbar):
                sig_mixed, sig_target, msg_bits, intrf_label, sinr_db = sample
                sig_mixed = sig_mixed.view(1, 1, sig_mixed.shape[-1]).float().to(device)
                sig_target = sig_target.view(1, 1, sig_target.shape[-1]).float().to(device)
                sig_pred = model(sig_mixed)

                sinr_db = int(sinr_db.item())
                idx_sinr = np.where(all_sinr_db == sinr_db)[0][0]
                mse_loss_model[intrf_label, idx_sinr] += criterion(sig_pred, sig_target).item()
                mse_loss[intrf_label, idx_sinr] += criterion(sig_mixed, sig_target).item()

                msg_bits = msg_bits.numpy()
                bits, _ = demodulator(split_to_complex(sig_mixed))
                bits_model, _ = demodulator(split_to_complex(sig_pred))
                ber[intrf_label, idx_sinr] += np.sum(msg_bits != bits) / msg_bits.shape[1]
                ber_model[intrf_label, idx_sinr] += np.sum(msg_bits != bits_model) / msg_bits.shape[1]

    mse_loss_model /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    mse_loss /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    ber_model /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    ber /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))

    return mse_loss_model, mse_loss, ber_model, ber


def split_to_complex(torch_tensor):
    # Splitting real and imaginary parts
    tf_tensor = tf.convert_to_tensor(torch_tensor.squeeze(0))
    real_part = tf_tensor[:, :tf_tensor.shape[1] // 2]
    imaginary_part = tf_tensor[:, tf_tensor.shape[1] // 2:]

    # Create complex tensor
    complex_tensor = tf.complex(real_part, imaginary_part)

    return complex_tensor
