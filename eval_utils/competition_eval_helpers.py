from comm_utils import demodulate_qpsk_signal, demodulate_ofdm_signal, split_to_complex
import torch
from tqdm import tqdm
import numpy as np


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

    with torch.no_grad():
        with tqdm(dataloader, desc='Evaluation', unit='Sample') as pbar:
            for i, sample in enumerate(pbar):
                sig_mixed, sig_target, msg_bits, intrf_label, sinr_db = sample
                sig_mixed, sig_target = sig_mixed.float().to(device), sig_target.float().to(device)
                sig_pred = model(sig_mixed)

                # Get SINR index
                sinr_db = int(sinr_db.item())
                idx_sinr = np.where(all_sinr_db == sinr_db)[0][0]

                # Convert to tf tensors for demodulation
                sig_mixed = split_to_complex(sig_mixed.detach().cpu())
                sig_pred = split_to_complex(sig_pred.detach().cpu())

                # Convert to numpy arrays for MSE calculation
                sig_target_numpy = split_to_complex(sig_target.detach().cpu()).numpy()
                sig_mixed_numpy = sig_mixed.numpy()
                sig_pred_numpy = sig_pred.numpy()
                msg_bits = msg_bits.detach().cpu().numpy()

                # Calculate MSE
                mse_loss_model[intrf_label, idx_sinr] += np.mean(np.abs(sig_pred_numpy - sig_target_numpy) ** 2)
                mse_loss[intrf_label, idx_sinr] += np.mean(np.abs(sig_mixed_numpy - sig_target_numpy) ** 2)

                # Calculate Ber
                bits, _ = demodulator(sig_mixed)
                bits_model, _ = demodulator(sig_target)
                ber[intrf_label, idx_sinr] += np.sum(msg_bits != bits) / msg_bits.shape[1]
                ber_model[intrf_label, idx_sinr] += np.sum(msg_bits != bits_model) / msg_bits.shape[1]

    mse_loss_model /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    mse_loss /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    ber_model /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    ber /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))

    return mse_loss_model, mse_loss, ber_model, ber
