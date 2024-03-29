import comm_utils
from tqdm import tqdm
import h5py
import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from datetime import datetime
from joblib import dump, load


# CONSTANTS
samples_per_symbol = 16
sig_len = 40_096
num_symbols = sig_len // samples_per_symbol
intrf_path_dir = Path('rf_datasets/train_set_unmixed/interference_set_frame/')
intrf_files = ['CommSignal2', 'CommSignal3', 'CommSignal5G1', 'EMISignal1']


def generate_train_mixture(soi_type, num_batches, batch_size):
    os.makedirs('datasets', exist_ok=True)
    dataset_path = Path(f'datasets/{soi_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(dataset_path)

    if soi_type == 'QPSK':
        gen_soi = comm_utils.generate_qpsk_signal
    elif soi_type == 'QPSK-OFDM':
        gen_soi = comm_utils.generate_ofdm_signal
    else:
        raise NotImplementedError

    intrf_frames = list()
    for file in intrf_files:
        with h5py.File(os.path.join(intrf_path_dir, file + '_raw_data.h5'), 'r') as data_h5file:
            intrf_frames.append(np.array(data_h5file.get('dataset')))

    num_batches //= len(intrf_frames)
    with tqdm(range(num_batches), desc='Data Generation', unit='batch') as pbar:
        for batch in pbar:
            sig_soi, _, _, _ = gen_soi(batch_size, num_symbols)

            # Generate random SINR values in decibels
            sinr_db = -31 * tf.random.uniform(shape=(batch_size, 1)) + 1
            gain_linear = tf.pow(10.0, -0.5 * sinr_db / 10)
            gain_complex = tf.complex(gain_linear, tf.zeros_like(gain_linear))
            phase = tf.random.uniform(shape=(batch_size, 1))
            phase_complex = tf.complex(phase, tf.zeros_like(phase))
            gain_phasor = gain_complex * tf.math.exp(1j * 2 * np.pi * phase_complex)

            for i, frame in enumerate(intrf_frames):
                sample_indices = np.random.randint(frame.shape[0], size=(batch_size,))
                frame = frame[sample_indices, :]
                snapshot_start_idx = np.random.randint(frame.shape[1] - sig_len, size=frame.shape[0])
                snapshot_indices = tf.cast(snapshot_start_idx.reshape(-1, 1)
                                           + np.arange(sig_len).reshape(1, -1), tf.int32)
                intrf_frame_snapshot = tf.experimental.numpy.take_along_axis(frame, snapshot_indices, axis=1)
                sig_mixed = sig_soi + gain_phasor * intrf_frame_snapshot

                # save batch
                sig_mixed = sig_mixed.numpy()
                sig_target = sig_soi.numpy()
                batch_data = [sig_mixed, sig_target, i]
                mixture_filename = f'{soi_type}_interf_{intrf_files[i]}_batch_{batch}'
                dump(batch_data, os.path.join(dataset_path, mixture_filename))
                del sig_mixed, sig_target, batch_data
