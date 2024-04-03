import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from comm_utils import split_to_complex_numpy
from pathlib import Path


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    batch_size = dataloader.batch_size

    with torch.no_grad():
        with tqdm(dataloader, desc='Validation', unit='batch') as pbar:
            for i, sample in enumerate(pbar):
                sig_mixed, sig_target = sample[0].float().to(device), sample[1].float().to(device)
                sig_pred = model(sig_mixed)
                loss = criterion(sig_pred, sig_target)

                # Update total loss
                total_loss += loss.item()

                # Update the progress bar with the current loss and accuracy
                pbar.set_postfix({'Val Loss': total_loss / ((i + 1) * batch_size)})

    return total_loss / (len(dataloader) * batch_size)


def visualize_results(model, dataloader, device, epoch, num_samples=3):
    def plot_complex_envelope(complex_sig: np.ndarray, label: str):
        plt.plot(np.abs(complex_sig), label='Envelope', color='m')
        plt.title(f'Envelope of {label}')
        plt.xlabel('timestep')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(Path(f'checkpoints/figures/epoch_{epoch}_{label}_{i}.png'))
        plt.show()
        plt.close()

    num_symbols = 16 * 20
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
            plot_complex_envelope(sig_mixed_numpy, 'mixture')
            plot_complex_envelope(sig_pred_numpy, 'prediction')
            plot_complex_envelope(sig_target_numpy, 'target')
