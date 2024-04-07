from tqdm import tqdm
import torch
import os


def train_epoch(model, dataloader, criterion, optimizer, device, input_bits=False):
    model.train()
    total_loss = 0.0
    batch_size = dataloader.batch_size

    with tqdm(dataloader, desc='Training', unit='batch') as pbar:
        for i, sample in enumerate(pbar):
            sig_mixed, sig_target = sample[0].float().to(device), sample[1].float().to(device)
            msg_bits = sample[2].float().to(device)
            optimizer.zero_grad()
            sig_pred = model(sig_mixed)
            if input_bits:
                loss = criterion(sig_pred, msg_bits)
            else:
                loss = criterion(sig_pred, sig_target)
            loss.backward()
            optimizer.step()

            # Update total loss
            total_loss += loss.item()

            # Update the progress bar with the current loss and accuracy
            pbar.set_postfix({'Train Loss': total_loss / ((i + 1) * batch_size)})

    return total_loss / (len(dataloader) * batch_size)


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, checkpoint_dir, prefix=''):
    checkpoint_info = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'validation_loss': val_loss
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'{prefix}model_epoch_{epoch}_val_loss_{val_loss:.4f}.pt')
    torch.save(checkpoint_info, checkpoint_path)
