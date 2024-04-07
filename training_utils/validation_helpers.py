import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from comm_utils import split_to_complex_numpy
from pathlib import Path


def evaluate_epoch(model, dataloader, criterion, device, input_bits=False):
    model.eval()
    total_loss = 0.0
    batch_size = dataloader.batch_size

    with torch.no_grad():
        with tqdm(dataloader, desc='Validation', unit='batch') as pbar:
            for i, sample in enumerate(pbar):
                sig_mixed, sig_target = sample[0].float().to(device), sample[1].float().to(device)
                msg_bits = sample[2].float().to(device)
                sig_pred = model(sig_mixed)
                if input_bits:
                    loss = criterion(sig_pred, msg_bits)
                else:
                    loss = criterion(sig_pred, sig_target)

                # Update total loss
                total_loss += loss.item()

                # Update the progress bar with the current loss and accuracy
                pbar.set_postfix({'Val Loss': total_loss / ((i + 1) * batch_size)})

    return total_loss / (len(dataloader) * batch_size)


def visualize_results(model, dataloader, device, epoch, num_samples=3, save=True):
    def plot_complex_envelope(sig_mixed, sig_pred, sig_target):
        figs, axs = plt.subplots(3, 1)
        axs[0].plot(np.abs(sig_mixed), color='b')
        axs[0].set_title(f'Envelope of mixture')
        axs[0].set_xlabel('timestep')
        axs[1].plot(np.abs(sig_pred), color='r')
        axs[1].set_title(f'Envelope of prediction')
        axs[1].set_xlabel('timestep')
        axs[2].plot(np.abs(sig_target), color='m')
        axs[2].set_title(f'Envelope of target')
        axs[2].set_xlabel('timestep')
        plt.tight_layout()
        if save:
            plt.savefig(Path(f'checkpoints/figures/epoch_{epoch}_sample_{i}.png'))
        plt.show()
        plt.close()

    num_symbols = 16 * 100
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break
            sig_mixed, sig_target = sample[0].float().to(device), sample[1].float()
            sig_pred = model(sig_mixed)

            # convert to complex numpy arrays
            sig_mixed_numpy = split_to_complex_numpy(sig_mixed[0])[:num_symbols]
            sig_pred_numpy = split_to_complex_numpy(sig_pred[0])[:num_symbols]
            sig_target_numpy = split_to_complex_numpy(sig_target[0])[:num_symbols]

            # plot input, prediction and target
            plot_complex_envelope(sig_mixed_numpy, sig_pred_numpy, sig_target_numpy)
