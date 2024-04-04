import argparse
from models import UNet
from dataset_utils import SigSepDataset
from torch.utils.data import DataLoader
from training_utils import train_epoch, evaluate_epoch, save_checkpoint, visualize_results
from torch import nn
from torch.optim import Adam, lr_scheduler
import torch
import os
from pathlib import Path


def main(args):
    # Load development dataset files
    dataset_dir = Path(args.dataset_dir)
    filepaths_list = [os.path.join(dataset_dir, batch_file) for batch_file in os.listdir(dataset_dir)]

    # Split into train and validation
    train_test_split = 0.8
    num_train_files = int(train_test_split * len(filepaths_list))
    train_set = SigSepDataset(filepaths_list[:num_train_files], dtype='real')
    val_set = SigSepDataset(filepaths_list[num_train_files:], dtype='real')

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize basic UNet model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = UNet()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=5e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5)

    # Training Loop
    num_epochs = 50
    checkpoint_dir = 'checkpoints'
    top_k_checkpoints = 3
    best_val_losses = [float('inf')] * top_k_checkpoints
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, 'figures'), exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {round(train_loss, 4)}, "
              f"Validation Loss: {round(val_loss, 4)}")
        visualize_results(model, val_loader, device, epoch + 1)
        # Check if the current validation loss is among the top k
        for i in range(top_k_checkpoints):
            if val_loss < best_val_losses[i]:
                best_val_losses.insert(i, val_loss)
                best_val_losses = best_val_losses[:top_k_checkpoints]  # Keep only the top k values
                save_checkpoint(epoch + 1, model, optimizer, train_loss, val_loss, checkpoint_dir, 'UNet_')
                break
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet model on a dataset')
    parser.add_argument('--dataset_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    args = parser.parse_args()
    main(args)
