import argparse
import os
import torch
from pathlib import Path
from models import UNet
from dataset_utils import SigSepDataset
from data_manipulation_utils import StandardScaler
from torch.utils.data import DataLoader
from training_utils import visualize_results
from eval_utils import plot_competition_figures, evaluate_competition_faster


def main(train_dataset_dir, dataset_dir, checkpoint_path, batch_size, num_workers):
    # fit standard scaler
    train_dataset_dir = Path(train_dataset_dir)
    filepaths_list = [os.path.join(train_dataset_dir, batch_file) for batch_file in os.listdir(train_dataset_dir)]
    train_val_split = 0.8
    num_train_files = int(train_val_split * len(filepaths_list))
    standard_scaler = StandardScaler(filepaths_list[:num_train_files])
    standard_scaler.fit()

    # load test set
    dataset_dir = Path(dataset_dir)
    filepaths_list = [os.path.join(dataset_dir, batch_file) for batch_file in os.listdir(dataset_dir)]
    test_set = SigSepDataset(filepaths_list, dtype='real')

    # Create dataloaders
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load basic UNet model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path)
    model = UNet()
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    visualize_results(model, test_loader, device, "", 10)
    intrf_sig_names, all_sinr_db, ber, mse_loss = evaluate_competition_faster([model,], test_loader,
                                                                              'QPSK', device)
    plot_competition_figures(intrf_sig_names, all_sinr_db, mse_loss[0], mse_loss[1], ber[0], ber[1], 'QPSK')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Evaluation Script")
    parser.add_argument("--train_dataset_dir", type=str, help="Path to the training dataset directory")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    args = parser.parse_args()
    main(args.train_dataset_dir, args.dataset_dir, args.checkpoint_path, args.batch_size, args.num_workers)