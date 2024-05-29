#!/usr/bin/python3
"""
A script to load the trained model and perform inference on the test data
and generates the competition evaluation results
"""

import torch
from omegaconf import OmegaConf
from src.DefaultTorchWaveNet import Wave
from models import GeneralUNet, WaveNet
from dataset_utils import generate_competition_eval_mixture
from dataset_utils import SigSepDataset
from eval_utils import evaluation_and_results2
from data_manipulation_utils import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import argparse
import os

intrf_signal_set = ['CommSignal2',
                    'CommSignal3', 'CommSignal5G1', 'EMISignal1']

parser = argparse.ArgumentParser(
    description="Perform inference on the Test1 data mixture for all models")
parser.add_argument('--baseline_model_path', type=str,
                    help='Path to the baseline model')
parser.add_argument('--num_models', type=int, default=1,
                    help='Number of models to evaluate')
parser.add_argument('--soi_type', type=str, default='QPSK',
                    help='Type of signal of interest (QPSK/QPSK_OFDM)')
parser.add_argument('--interference_type', type=str, default='CommSignal2',
                    help='Type of interference signal (CommSignal2/CommSignal3/CommSignal5G1/EMISignal1/all)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for inference (number of examples per batch)')
parser.add_argument('--interference_dir_path', type=str,
                    default="rf_datasets/train_test_set_unmixed/datasets/testset1_frame/",
                    help='Path to the interference signals directory')


def main():
    args = parser.parse_args()

    if not args.baseline_model_path:
        raise ValueError("Baseline model path not provided")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # Load the model
    cfg = OmegaConf.load('src/configs/wavenet.yaml')
    model_baseline = Wave(cfg.model).to(device)
    model_baseline.load_state_dict(torch.load(
        args.baseline_model_path).get('model', None))
    if model_baseline is None:
        raise ValueError("Model could not be loaded")

    model_list = [{
        'model': model_baseline,
        'model_name': 'Baseline model',
        "standardized": False
    }]

    if args.num_models > 2:
        raise NotImplementedError(
            "Only 2 models can be evaluated at this time")
    elif args.num_models == 2:
        unet_model_path = input("Enter the path to the UNet model: ")
        wavenet_model_path = input("Enter the path to the WaveNet model: ")
        # Load the models
        loaded_path = torch.load(unet_model_path)
        model_unet = GeneralUNet().to(device)
        model_unet.load_state_dict(loaded_path.get(
            'model_state_dict', loaded_path.get('state_dict', loaded_path)))
        model_list.append({
            'model': model_unet,
            'model_name': 'UNet model',
            "standardized": True
        })
        loaded_path = torch.load(wavenet_model_path)
        model_params = loaded_path.get('model_params', {
            'input_channels': 2,
            'residual_channels': 256,
            'residual_layers': 30,
            'dilation_cycle_length': 10
        })
        model_wavenet = WaveNet(**model_params).to(device)
        model_wavenet.load_state_dict(loaded_path.get(
            'state_dict', loaded_path.get('model_state_dict', loaded_path)))
        model_list.append({
            'model': model_wavenet,
            'model_name': 'WaveNet model',
            "standardized": False
        })
    elif args.num_models == 1:
        type_of_model = input("Enter the type of model (UNet/WaveNet): ")
        if type_of_model == 'UNet':
            unet_model_path = input("Enter the path to the UNet model: ")
            # Load the model
            loaded_path = torch.load(unet_model_path)
            model_unet = GeneralUNet().to(device)
            model_unet.load_state_dict(loaded_path.get(
                'model_state_dict', loaded_path.get('state_dict', loaded_path)))
            model_list.append({
                'model': model_unet,
                'model_name': 'UNet model',
                "standardized": True
            })
        elif type_of_model == 'WaveNet':
            wavenet_model_path = input("Enter the path to the WaveNet model: ")
            # load the model
            loaded_path = torch.load(wavenet_model_path)
            model_params = loaded_path.get('model_params', {
                'input_channels': 2,
                'residual_channels': 256,
                'residual_layers': 30,
                'dilation_cycle_length': 10
            })
            model_wavenet = WaveNet(**model_params).to(device)
            model_wavenet.load_state_dict(loaded_path.get(
                'state_dict', loaded_path.get('model_state_dict', loaded_path)))
            model_list.append({
                'model': model_wavenet,
                'model_name': 'WaveNet model',
                "standardized": False
            })
        else:
            raise ValueError("Invalid model type")

    model_names = ""
    for i, model in enumerate(model_list):
        model_names += f"{i+1}. {model['model_name']} -> {sum(p.numel() for p in model['model'].parameters()) / 1e6:.2f} M\n"

    print(
        f"\n\nInference will be performed on the Test1 data mixture for the following models:\n{model_names}\nThey will be evaluated for the {args.soi_type} on the following interference signals: {','.join(intrf_signal_set) if args.interference_type == 'all' else args.interference_type}\n\n")

    train_filepath = "rf_datasets/train_set_mixed/datasets/QPSK_20240407_211116" if args.soi_type == "QPSK" else "rf_datasets/train_set_mixed/datasets/QPSK_OFDM_20240407_211404"
    train_filepaths_list = [os.path.join(train_filepath, batch_file)
                            for batch_file in os.listdir(train_filepath)]
    train_size = int(0.8 * len(train_filepaths_list))
    scaler = StandardScaler(train_filepaths_list[:train_size])
    if args.soi_type == 'QPSK':
        scaler.mu = -0.00023688253749715093 + 1j * 0.000709482444289711
        scaler.sigma = 7.347274835527751
    elif args.soi_type == 'QPSK_OFDM':
        scaler.mu = -0.00040491584097022406 + 1j * 0.000572398170269466
        scaler.sigma = 7.160499862913227

    mixed_dataset_path, _, _ = generate_competition_eval_mixture(
        args.soi_type, args.interference_dir_path)

    # load the dataset
    filepaths_list = [os.path.join(mixed_dataset_path, batch_file)
                      for batch_file in os.listdir(mixed_dataset_path)]
    dataset_standardized = SigSepDataset(filepaths_list, scaler, dtype="real")
    dataset = SigSepDataset(filepaths_list, dtype="real")
    dataloader_standardized = torch.utils.data.DataLoader(
        dataset_standardized, batch_size=args.batch_size, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False)

    # Perform inference
    models, model_names, standardized = zip(*[(model['model'], model['model_name'], model['standardized'])
                                              for model in model_list])
    evaluation_and_results2(models, (dataloader, dataloader_standardized),
                            args.soi_type, args.interference_type, device, model_names, standardized)


if __name__ == "__main__":
    main()
