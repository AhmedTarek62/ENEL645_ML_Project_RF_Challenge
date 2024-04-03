import torch
from tqdm import tqdm


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
