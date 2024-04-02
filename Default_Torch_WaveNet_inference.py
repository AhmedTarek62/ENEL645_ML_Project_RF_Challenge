#!/usr/bin/python3
"""
A script to load the trained model and perform inference on the test data
It also plots the MSE and BER curves
"""

import torch
from src.Default_Torch_WaveNet import Wave
from omegaconf import OmegaConf
from dataset_utils import generate_competition_eval_mixture
from dataset_utils import SigSepDataset
from torch.utils.data import DataLoader
from eval_utils import postprocessing_helpers
import numpy as np
from tqdm import tqdm
import argparse

# constants
intrf_files = ['CommSignal2', 'CommSignal3', 'CommSignal5G1', 'EMISignal1']


def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\nUsing device: {device}")

    # load the model
    cfg = OmegaConf.load("src/configs/wavenet.yaml")
    model = Wave(cfg.model).to(device)
    model.load_state_dict(torch.load(kwargs['model_path'])["model"])

    print(
        f"The model has {sum(p.numel() for p in model.parameters())/1e6} million parameters")

    # load the dataset
    dataset = SigSepDataset(kwargs['dataset_path'], dtype="real")
    dataloader = DataLoader(
        dataset, batch_size=kwargs['batch_size'], shuffle=False)
    batch = next(iter(dataloader))
    soi_mix, soi_target, msg_bits, intrf_labels, sinr_db = batch
    # print(soi_mix.shape, soi_target.shape, msg_bits.shape, intrf_labels.shape, sinr_db.shape)

    model.eval()

    sinr_all = []
    msg_bits_all = []
    soi_target_all = []
    soi_no_mitigation_all = []
    soi_est_all = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc='Evaluation', total=kwargs['num_batches']):
            soi_mix, soi_target, msg_bits, intrf_labels, sinr_db = batch
            # only process for one interference type
            idx = intrf_files.index(kwargs['interference_type'])
            soi_mix, soi_target = soi_mix[intrf_labels == idx].to(
                device), soi_target[intrf_labels == idx].to(device)
            sinr_db = sinr_db[intrf_labels == idx]
            msg_bits = msg_bits[intrf_labels == idx]

            soi_est = model(soi_mix)

            soi_est_all.extend(soi_est.cpu().numpy())
            soi_no_mitigation_all.extend(soi_mix.cpu().numpy())
            soi_target_all.extend(soi_target.cpu().numpy())
            sinr_all.extend(sinr_db.cpu().numpy())
            msg_bits_all.extend(msg_bits.cpu().numpy())

        soi_est_all = np.array(soi_est_all)
        soi_no_mitigation_all = np.array(soi_no_mitigation_all)
        soi_target_all = np.array(soi_target_all)
        sinr_all = np.array(sinr_all)
        msg_bits_all = np.array(msg_bits_all)
        # print(soi_est_all.shape, soi_no_mitigation_all.shape,
        #       soi_target_all.shape, sinr_all.shape, msg_bits_all.shape)

        # combine IQ channels
        soi_no_mitigation_all = soi_no_mitigation_all[:,
                                                      0, :] + 1j * soi_no_mitigation_all[:, 1, :]
        soi_target_all = soi_target_all[:, 0, :] + 1j * soi_target_all[:, 1, :]
        soi_est_all = soi_est_all[:, 0, :] + 1j * soi_est_all[:, 1, :]

        # postprocessing
        if kwargs['soi_type'] == 'QPSK':
            results = postprocessing_helpers.postprocess_qpsk(
                soi_est_all, soi_no_mitigation_all, soi_target_all, msg_bits_all, sinr_all)
        elif kwargs['soi_type'] == 'QPSK_OFDM':
            results = postprocessing_helpers.postprocess_ofdm(
                soi_est_all, soi_no_mitigation_all, soi_target_all, msg_bits_all, sinr_all)

        # visualize the results
        postprocessing_helpers.visualize_results(results, kwargs['soi_type'],
                                                 kwargs['interference_type'], model_name="Default_Torch_WaveNet")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform inference on the test1 data mixture')
    parser.add_argument('--model_path', type=str,
                        default='torchmodels/model.pth', help='Path to the trained model')
    parser.add_argument('--soi_type', type=str, default='QPSK',
                        help='Type of signal of interest (QPSK/QPSK_OFDM)')
    parser.add_argument('--interference_type', type=str, default='CommSignal2',
                        help='Type of interference signal (CommSignal2/CommSignal3/CommSignal5G1/EMISignal1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (number of examples per batch)')
    parser.add_argument('--interference_dir_path', type=str,
                        default="rf_datasets/train_test_set_unmixed/datasets/testset1_frame/", help='Number of epochs to train the model')
    args = parser.parse_args()

    dataset_path, num_batches, batch_size = generate_competition_eval_mixture(
        args.soi_type, args.interference_dir_path)
    args_dict = vars(args)
    args_dict['dataset_path'] = dataset_path
    args_dict['num_batches'] = (num_batches * batch_size) // args.batch_size
    main(**args_dict)
