from comm_utils import demodulate_qpsk_signal, demodulate_ofdm_signal, split_to_complex, split_to_complex_batch
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
                bits_model, _ = demodulator(sig_pred)
                ber[intrf_label, idx_sinr] += np.sum(msg_bits != bits) / bits.shape[0]
                ber_model[intrf_label, idx_sinr] += np.sum(msg_bits != bits_model) / bits.shape[0]

    mse_loss_model /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    mse_loss /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    ber_model /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    ber /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))

    return intrf_sig_names, all_sinr_db, mse_loss_model, mse_loss, ber_model, ber


def evaluate_competition_fast(model, dataloader, soi_type, device,
                               intrf_sig_names=('CommSignal2', 'CommSignal3', 'CommSignal5G1', 'EMISignal1')):
    if soi_type == 'QPSK':
        demodulator = demodulate_qpsk_signal
    elif soi_type == 'QPSK_OFDM':
        demodulator = demodulate_ofdm_signal
    else:
        raise NotImplementedError

    all_sinr_db = np.arange(-30, 1, 3)
    mse_loss = np.zeros((len(intrf_sig_names), len(all_sinr_db)))
    ber = np.zeros((len(intrf_sig_names), len(all_sinr_db)))
    if model:
        model = model.to(device)
        model.eval()

    with torch.no_grad():
        with tqdm(dataloader, desc='Evaluation', unit='Sample') as pbar:
            for i, batch in enumerate(pbar):
                sig_mixed_batch, sig_target_batch, msg_bits_batch, intrf_label_batch, sinr_db_batch = batch
                sig_mixed_batch = sig_mixed_batch.float().to(device)
                sig_target_batch = sig_target_batch.float().to(device)
                if model:
                    sig_pred_batch = model(sig_mixed_batch)
                else:
                    sig_pred_batch = sig_mixed_batch  # no mitigation case

                # Get interference and SINR indices
                sinr_db_batch = sinr_db_batch.detach().cpu().numpy()
                sinr_indices = np.array([np.where(all_sinr_db == sinr_db)[0][0] for sinr_db in sinr_db_batch])
                intrf_indices = intrf_label_batch.detach().cpu().numpy()

                # Convert to tf tensors for demodulation
                sig_pred_batch = split_to_complex_batch(sig_pred_batch.detach().cpu())
                sig_target_batch = split_to_complex_batch(sig_target_batch.detach().cpu())
                # Convert to numpy arrays for MSE calculation
                sig_target_numpy = sig_target_batch.numpy()
                sig_pred_numpy = sig_pred_batch.numpy()
                msg_bits_numpy = msg_bits_batch.detach().cpu().numpy()

                # Calculate MSE
                mse_batch = np.mean(np.abs(sig_pred_numpy - sig_target_numpy) ** 2, axis=1)
                for sample_idx, (intrf_idx, sinr_idx) in enumerate(zip(intrf_indices, sinr_indices)):
                    mse_loss[intrf_idx, sinr_idx] += mse_batch[sample_idx]

                # Calculate Ber
                bits, _ = demodulator(sig_pred_batch)
                ber_batch = np.sum(msg_bits_numpy != bits, axis=1) / bits.shape[1]
                for sample_idx, (intrf_idx, sinr_idx) in enumerate(zip(intrf_indices, sinr_indices)):
                    ber[intrf_idx, sinr_idx] += ber_batch[sample_idx]

    mse_loss /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))
    ber /= (len(dataloader) / len(intrf_sig_names) / len(all_sinr_db))

    return ber, mse_loss


def evaluate_competition_faster(model_list, dataloader, soi_type, device,
                               intrf_sig_names=('CommSignal2', 'CommSignal3', 'CommSignal5G1', 'EMISignal1')):
    if soi_type == 'QPSK':
        demodulator = demodulate_qpsk_signal
    elif soi_type == 'QPSK_OFDM':
        demodulator = demodulate_ofdm_signal
    else:
        raise NotImplementedError

    all_sinr_db = np.arange(-30, 1, 3)
    mse_loss = np.zeros((len(model_list) + 1, len(intrf_sig_names), len(all_sinr_db)))
    ber = np.zeros((len(model_list) + 1, len(intrf_sig_names), len(all_sinr_db)))

    model_list.append(None)
    for model in model_list:
        if model:
            model = model.to(device)
            model.eval()

    with torch.no_grad():
        with tqdm(dataloader, desc='Evaluation', unit='Sample') as pbar:
            for i, batch in enumerate(pbar):
                sig_mixed_batch, sig_target_batch, msg_bits_batch, intrf_label_batch, sinr_db_batch = batch
                sig_mixed_batch = sig_mixed_batch.float().to(device)
                sig_target_batch = sig_target_batch.float().to(device)

                # inference for all models
                sig_pred_batch_list = list()
                for model in model_list:
                    if model:
                        sig_pred_batch = model(sig_mixed_batch)
                    else:
                        sig_pred_batch = sig_mixed_batch  # no mitigation case
                    sig_pred_batch_list.append(sig_pred_batch)

                # Get interference and SINR indices
                sinr_db_batch = sinr_db_batch.detach().cpu().numpy()
                sinr_indices = np.array([np.where(all_sinr_db == sinr_db)[0][0] for sinr_db in sinr_db_batch])
                intrf_indices = intrf_label_batch.detach().cpu().numpy()

                # Convert to tf tensors for demodulation
                for sample_idx, sig_pred_batch in enumerate(sig_pred_batch_list):
                    sig_pred_batch_list[sample_idx] = split_to_complex_batch(sig_pred_batch.detach().cpu())
                sig_target_batch = split_to_complex_batch(sig_target_batch.detach().cpu())

                # Convert to numpy arrays for MSE calculation
                sig_pred_batch_numpy_list = list()
                for sig_pred_batch in sig_pred_batch_list:
                    sig_pred_batch_numpy_list.append(sig_pred_batch.numpy())

                sig_target_batch_numpy = sig_target_batch.numpy()
                msg_bits_numpy = msg_bits_batch.detach().cpu().numpy()

                # Calculate MSE
                for model_idx in range(len(model_list)):
                    mse_batch = np.mean(
                        np.abs(sig_pred_batch_numpy_list[model_idx] - sig_target_batch_numpy) ** 2, axis=1)
                    for sample_idx, (intrf_idx, sinr_idx) in enumerate(zip(intrf_indices, sinr_indices)):
                        mse_loss[model_idx, intrf_idx, sinr_idx] += mse_batch[sample_idx]

                # Calculate Ber
                for model_idx, sig_pred_batch in enumerate(sig_pred_batch_list):
                    bits, _ = demodulator(sig_pred_batch)
                    ber_batch = np.sum(msg_bits_numpy != bits, axis=1) / bits.shape[1]
                    for sample_idx, (intrf_idx, sinr_idx) in enumerate(zip(intrf_indices, sinr_indices)):
                        ber[model_idx, intrf_idx, sinr_idx] += ber_batch[sample_idx]

    mse_loss /= (len(dataloader) * dataloader.batch_size / len(intrf_sig_names) / len(all_sinr_db))
    ber /= (len(dataloader) * dataloader.batch_size / len(intrf_sig_names) / len(all_sinr_db))

    return intrf_sig_names, all_sinr_db, ber, mse_loss
