#!/usr/bin/python3

import os
import torch
from models import WaveNet
from dataset_utils import SigSepDataset
from dataset_utils import generate_competition_eval_mixture
from torch.utils.data import DataLoader
from training_utils import visualize_results
from eval_utils import evaluate_competition, evaluation_and_results, plot_competition_figures


import argparse

# CONSTANTS
intrf_signal_set = ["CommSignal2",
                    "CommSignal3", "CommSignal5G1", "EMISignal1"]


def main(**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\nUsing device: {device}")

    # load the model
    loaded_model = torch.load(kwargs['ckpt_path'])
    
    model_params = loaded_model.get("model_params", {
        "input_channels": 2,
        "residual_channels": 256,
        "residual_layers": 30,
        "dilation_cycle_length": 10
    })
    model = WaveNet(**model_params).to(device)
    model.load_state_dict(loaded_model.get("state_dict", loaded_model.get("model_state_dict", loaded_model)))
    
    print(
        f"The model has {sum(p.numel() for p in model.parameters())/1e6} million parameters")

    # load the dataset
    filepaths_list = [os.path.join(kwargs['dataset_dir'], batch_file)
                      for batch_file in os.listdir(kwargs['dataset_dir'])]
    dataset = SigSepDataset(filepaths_list, dtype="real")
    dataloader = DataLoader(
        dataset, batch_size=kwargs['batch_size'], shuffle=False)

    # visualize_results(model, dataloader, device, "", 10)
    
    evaluation_and_results(model, dataloader, kwargs["soi_type"], device, "KUTII_WaveNet")
    # intrf_sig_names, all_sinr_db, mse_loss_model, mse_loss, ber_model, ber = (
    #     evaluate_competition(model, dataloader, kwargs['soi_type'], device))
    # plot_competition_figures(intrf_sig_names, all_sinr_db,
    #                          mse_loss_model, mse_loss, ber_model, ber, kwargs['soi_type'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='KUTII_WaveNet evaluation script')
    parser.add_argument('--ckpt_path', type=str,
                        help='Path to the model checkpoint')
    parser.add_argument('--dataset_dir', type=str,
                        help='Path to the dataset directory')
    parser.add_argument('--soi_type', type=str,
                        help='Type of signal of interest (QPSK/QPSK_OFDM)', default='QPSK')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--intrf_dataset_path', type=str, help='Path to the interference dataset',
                        default="rf_datasets/train_test_set_unmixed/dataset/testset1_frame")
    args = parser.parse_args()
    if not args.ckpt_path:
        raise ValueError("Please provide a checkpoint path")
    if not args.dataset_dir:
        args.dataset_dir, _, _ = generate_competition_eval_mixture(
            args.soi_type, args.intrf_dataset_path)
    main(**vars(args))
