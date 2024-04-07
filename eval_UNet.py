import argparse
import os
import torch
from pathlib import Path
from models import UNet, GeneralUNet
from dataset_utils import SigSepDataset
from data_manipulation_utils import StandardScaler, RangeScaler
from torch.utils.data import DataLoader
from training_utils import visualize_results
from eval_utils import plot_competition_figures, evaluate_competition_faster


def main(train_dataset_dir, dataset_dir, checkpoint_path, batch_size, num_workers):

    train_dataset_dir = Path(train_dataset_dir)
    filepaths_list = [os.path.join(train_dataset_dir, batch_file) for batch_file in os.listdir(train_dataset_dir)]
    train_val_split = 0.8
    num_train_files = int(train_val_split * len(filepaths_list))

    # Fit the scaler
    if args.preprocess == 'standard':
        scaler = StandardScaler(filepaths_list[:num_train_files])
        scaler.fit()
    elif args.preprocess == 'range':
        scaler = RangeScaler(filepaths_list[:num_train_files])
        scaler.fit()
    else:
        scaler = None

    # load test set
    dataset_dir = Path(dataset_dir)
    filepaths_list = [os.path.join(dataset_dir, batch_file) for batch_file in os.listdir(dataset_dir)]
    test_set = SigSepDataset(filepaths_list, scaler, dtype='real')

    # Create dataloaders
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load  UNet model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path)

    if args.model == 'UNet':
        model = UNet()
    elif args.model == 'GeneralUNet':
        model = GeneralUNet()
    else:
        raise NotImplementedError
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    visualize_results(model, test_loader, device, "", 10)
    intrf_sig_names, all_sinr_db, ber, mse_loss = evaluate_competition_faster([model,], test_loader,
                                                                              'QPSK', device)
    plot_competition_figures(intrf_sig_names, all_sinr_db, mse_loss[0], mse_loss[1], ber[0], ber[1], 'QPSK')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNet Evaluation Script")
    parser.add_argument('--model', type=str, default='GeneralUNet', choices=['GeneralUNet', 'UNet'],
                        help='Model name')
    parser.add_argument('--preprocess', type=str, options=['standard', 'range', 'none'],
                        help='Type of preprocessing to use')
    parser.add_argument("--train_dataset_dir", type=str, help="Path to the training dataset directory")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument('--prefix', type=str, default='UNet_', help='Prefix for saving checkpoints')
    args = parser.parse_args()
    main(args.train_dataset_dir, args.dataset_dir, args.checkpoint_path, args.batch_size, args.num_workers)
