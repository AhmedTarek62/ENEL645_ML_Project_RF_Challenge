#!/usr/bin/python3
"""
trainer script for KUTII_WaveNet  model
"""

import os
import torch
from dataset_utils import generate_train_mixture
from dataset_utils import SigSepDataset
from data_manipulation_utils import StandardScaler
from torch.utils.data import DataLoader
from models import WaveNet

from comm_utils import demodulate_qpsk_signal, demodulate_ofdm_signal, create_resource_grid

from training_utils import SoftDemodLoss, BerLoss

from tqdm import tqdm

import numpy as np

import wandb


import argparse

# CONSTANTS
intrf_signal_set = ['CommSignal2',
                    'CommSignal3', 'CommSignal5G1', 'EMISignal1']
sig_len = 40_960


def train_loop(model, train_loader, epoch, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for (soi_mix, soi_target, _, _, _) in tqdm(train_loader, desc=f'Training: [Epoch {epoch + 1}]', unit='batch'):
        optimizer.zero_grad()
        soi_mix, soi_target = soi_mix.to(device), soi_target.to(device)
        soi_est = model(soi_mix)
        loss = criterion(soi_est, soi_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validation_loop(model, val_loader, epoch, criterion, device, callback=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for (soi_mix, soi_target, msg_bits, intrf_type, sinr) in tqdm(val_loader, desc=f'Validation: [Epoch {epoch + 1}]', unit='batch'):
            soi_mix, soi_target = soi_mix.to(device), soi_target.to(device)

            soi_est = model(soi_mix)
            loss = criterion(soi_est, soi_target)
            if callback is not None:
                callback(soi_est.cpu().detach(), soi_target.cpu().detach(), dict(
                    zip(msg_bits, intrf_type, sinr)))
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main(**kwargs):
    epochs = kwargs["epochs"]
    batch_size = kwargs["batch_size"]
    lr = kwargs["lr"]
    optimizer = kwargs["optimizer"]
    dataset_path = kwargs["dataset_path"]
    loss = kwargs['loss']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\nUsing device: {device}")

    # load and split the dataset
    if not kwargs["preprocess"]:
        filepaths_list = [os.path.join(dataset_path, batch_file)
                        for batch_file in os.listdir(dataset_path)]
        dataset = SigSepDataset(filepaths_list, dtype="real")
        total_samples = len(dataset)
        train_samples = int(0.8 * total_samples)
        val_samples = total_samples - train_samples
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_samples, val_samples])
    else:
        filepaths_list = [os.path.join(dataset_path, batch_file)
                            for batch_file in os.listdir(dataset_path)]
        np.random.shuffle(filepaths_list)
        train_size = int(0.8 * len(filepaths_list))
        train_filepaths_list = filepaths_list[:train_size]
        standard_scaler = StandardScaler(train_filepaths_list)
        standard_scaler.fit()
        train_dataset = SigSepDataset(train_filepaths_list, standard_scaler, dtype='real')
        val_dataset = SigSepDataset(filepaths_list[train_size:], standard_scaler, dtype='real')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    model_params = {
        "input_channels": 2,
        "residual_channels": 256,
        "residual_layers": 30,
        "dilation_cycle_length": 10
    }

    model = WaveNet(**model_params).to(device)
    print("The model has", sum(p.numel()
          for p in model.parameters())/1e6, "million parameters")

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.01)

    if loss == "mse": 
        criterion = torch.nn.MSELoss()
    else:
        demodulator = demodulate_qpsk_signal if kwargs["soi_type"] == "QPSK" else demodulate_ofdm_signal
        resource_grid = None if kwargs["soi_type"] == "QPSK" else create_resource_grid(sig_len//80)
        if kwargs["loss"] == "softdemod":
            criterion = SoftDemodLoss(demodulator, device, resource_grid)
        elif kwargs["loss"] == "ber":
            criterion = BerLoss(demodulator, device, resource_grid)
        else:
            raise NotImplemented

    run = wandb.init(
        project="Signal Separation Challenge ICASSP 2024", config=kwargs)

    top_k_checkpoints = 3
    best_val_loss = [float("inf")] * top_k_checkpoints

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    for epoch in range(epochs):
        avg_tloss = train_loop(
            model, train_loader, epoch, optimizer, criterion, device)
        avg_vloss = validation_loop(
            model, val_loader, epoch, criterion, device)

        print(f"\tTrain Loss: {avg_tloss}")
        print(f"\tValidation Loss: {avg_vloss}")

        run.log({"train_loss": avg_tloss, "val_loss": avg_vloss})

        if avg_vloss < max(best_val_loss):
            max_best_val_loss = max(best_val_loss)
            idx = best_val_loss.index(max_best_val_loss)
            best_val_loss[idx] = avg_vloss
            best_model_path = f"checkpoints/KUTII_WaveNet_{kwargs['soi_type']}-{epoch+1}_{avg_vloss}.pt"
            torch.save({"model_state_dict": model.state_dict(),
                        "model_params": model_params}, best_model_path)
            print(f"Model saved at {best_model_path}")

            for file in os.listdir("checkpoints"):
                if file.endswith(f"_{max_best_val_loss}.pt"):
                    os.remove(f"checkpoints/{file}")

        scheduler.step()
    wandb.config.update({
        "best_model_path": best_model_path
    })
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='KUTII_WaveNet Trainer Script')
    parser.add_argument('--soi_type', type=str, default='QPSK',
                        help="Type of signal of interest (QPSK/QPSK_OFDM)")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--preprocess', type=str, default=None)
    parser.add_argument('--num_batches', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        help="Optimizer to use (adam/adamw)")
    parser.add_argument('--loss', type=str, default='mse',
                        help="Loss function to use (mse/softdemod/ber)")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--intrf_dataset_path", type=str,
                        default="rf_datasets/train_test_set_unmixed/dataset/interferenceset_frame")

    args = parser.parse_args()
    if not args.dataset_path:
        args.dataset_path = generate_train_mixture(
            soi_type=args.soi_type, num_batches=args.num_batches, batch_size=args.batch_size, intrf_path_dir=args.intrf_dataset_path)
    main(**vars(args))
