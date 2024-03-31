#!/usr/bin/python3
"""
A script to load the trained model and perform inference on the test data
It also plots the MSE and BER curves
"""

import torch
from src.Default_Torch_WaveNet import Wave
from omegaconf import OmegaConf
from dataset_utils.generate_train_mixture import generate_train_mixture
from dataset_utils import SigSepDataset
from eval_utils import postprocessing_helpers

import argparse


def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the model
    cfg = OmegaConf.load("src/configs/wavenet.yaml")
    model = Wave(cfg.model).to(device)
    model.load_state_dict(torch.load(kwargs['model_path'])["model"])

    print(
        f"The model has {sum(p.numel() for p in model.parameters())/1e6} million parameters")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Perform inference on the test data')
    parser.add_argument('--model_path', type=str,
                        default='torchmodels/model.pth', help='Path to the trained model')
    parser.add_argument('--soi_type', type=str, default='QPSK',
                        help='Type of signal of interest (QPSK/QPSK_OFDM)')
    parser.add_argument('--num_batches', type=int, default=50,
                        help='Number of batches to generate')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size (number of examples per batch)')
    parser.add_argument('--dataset', type=str, default='test',
                        help='Dataset to generate (train/test)')
    parser.add_argument('--interference_dir_path', type=str,
                        default="rf_datasets/train_test_set_unmixed/datasets/testset1_frame/", help='Number of epochs to train the model')
    args = parser.parse_args()

    # generate_train_mixture(args.soi_type, args.num_batches,
    #                        args.batch_size, args.dataset, args.interference_dir_path)
    main(**vars(args))
