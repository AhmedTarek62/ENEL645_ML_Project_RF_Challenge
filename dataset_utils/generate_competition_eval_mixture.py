import comm_utils
from tqdm import tqdm
import h5py
import os
import numpy as np
from pathlib import Path
import tensorflow as tf
from datetime import datetime
from joblib import dump


samples_per_symbol = 16
ofdm_symbol_len = 80  # Cyclic Prefix (16) + Subcarriers (64)
sig_len = 40_960
intrf_files = ['CommSignal2', 'CommSignal3', 'CommSignal5G1', 'EMISignal1']


def generate_competition_eval_mixture(soi_type,
                                      intrf_path_dir=Path('rf_datasets/train_set_unmixed/interference_set_frame/')):
    """
    Generate competition eval set which 100 test cases per target SINR level, ranging from -30 dB to 0 dB at 3 dB
    steps, with 100 test cases per target SINR level and interference signal.
    :param soi_type:
    :param intrf_path_dir:
    :return:
    """

    dataset_path = Path(
        f'rf_datasets/test_set_mixed/datasets/eval_{soi_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(dataset_path)

    if soi_type == 'QPSK':
        gen_soi = comm_utils.generate_qpsk_signal
        num_symbols = sig_len // samples_per_symbol
        bits_per_stream = 2 * num_symbols
    elif soi_type == 'QPSK_OFDM':
        gen_soi = comm_utils.generate_ofdm_signal
        num_symbols = sig_len // ofdm_symbol_len
        bits_per_stream = 2 * 56 * num_symbols
    else:
        raise NotImplementedError

    intrf_frames = list()
    for file in intrf_files:
        with h5py.File(os.path.join(intrf_path_dir, file + '_test1_raw_data.h5'), 'r') as data_h5file:
            intrf_frames.append(np.array(data_h5file.get('dataset')))

    all_sinr_db = np.arange(-30, 1, 3)
    num_test_cases = 100
    num_intrf_signals = len(intrf_files)
    intrf_labels = np.array([i for i in range(len(intrf_frames)) for _ in range(num_test_cases)])
    num_batches = len(all_sinr_db)

    for sinr_db in tqdm(all_sinr_db, desc='Generating Evaluation Data', unit='SINR Value'):
        sig_soi, msg_bits = gen_soi(num_test_cases, num_symbols)
        padding = [[0, 0], [0, sig_len - sig_soi.shape[1]]]
        sig_soi = tf.pad(sig_soi, padding, "CONSTANT")
        gain_linear = tf.pow(10.0, -0.5 * sinr_db / 10)
        gain_complex = tf.complex(gain_linear, tf.zeros_like(gain_linear))
        phase = tf.random.uniform(shape=(num_test_cases, 1))
        phase_complex = tf.complex(phase, tf.zeros_like(phase))
        gain_phasor = gain_complex * tf.math.exp(1j * 2 * np.pi * phase_complex)
        sig_mixed_numpy = np.zeros((num_test_cases * num_intrf_signals, sig_len), dtype=complex)
        sig_soi_numpy = np.zeros((num_test_cases * num_intrf_signals, sig_len), dtype=complex)
        msg_bits_numpy = np.zeros((num_test_cases * num_intrf_signals, bits_per_stream))

        for i, frame in enumerate(intrf_frames):
            sample_indices = np.random.randint(frame.shape[0], size=(num_test_cases,))
            frame = frame[sample_indices, :]
            snapshot_start_idx = np.random.randint(frame.shape[1] - sig_len, size=frame.shape[0])
            snapshot_indices = tf.cast(snapshot_start_idx.reshape(-1, 1)
                                       + np.arange(sig_len).reshape(1, -1), tf.int32)
            intrf_frame_snapshot = tf.experimental.numpy.take_along_axis(frame, snapshot_indices, axis=1)
            sig_mixed = sig_soi + gain_phasor * intrf_frame_snapshot
            sig_mixed_numpy[i * num_test_cases: (i + 1) * num_test_cases, :] = sig_mixed.numpy()
            sig_soi_numpy[i * num_test_cases: (i + 1) * num_test_cases, :] = sig_soi.numpy()
            msg_bits_numpy[i * num_test_cases: (i + 1) * num_test_cases, :] = msg_bits.numpy()
            del sig_mixed

        # save SINR case
        sinr_db_numpy = sinr_db * np.ones((num_test_cases * num_intrf_signals))
        batch_data = [sig_mixed_numpy, sig_soi_numpy, msg_bits_numpy, intrf_labels, sinr_db_numpy]
        mixture_filename = f'{soi_type}_sinr_{sinr_db}'
        dump(batch_data, os.path.join(dataset_path, mixture_filename))

    print(f'\nDataset saved at {dataset_path}')
    return dataset_path, num_batches, num_test_cases * num_intrf_signals
