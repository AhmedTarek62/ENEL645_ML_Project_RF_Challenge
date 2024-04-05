from models import UNet
from dataset_utils import SigSepDataset
from torch.utils.data import DataLoader
from training_utils import visualize_results
from eval_utils import plot_competition_figures, evaluate_competition_faster
import torch
import os
from pathlib import Path


# load test set
dataset_dir = 'rf_datasets/test_set_mixed/datasets/eval_QPSK_20240405_054720'
dataset_dir = Path(dataset_dir)
filepaths_list = [os.path.join(dataset_dir, batch_file) for batch_file in os.listdir(dataset_dir)]
test_set = SigSepDataset(filepaths_list, dtype='real')

# Create dataloaders
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

# Load basic UNet model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
ckpt_path = Path('checkpoints/UNet_model_epoch_37_val_loss_0.0037.pt')
ckpt = torch.load(ckpt_path)
model = UNet()
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device)
visualize_results(model, test_loader, device, "", 10)
intrf_sig_names, all_sinr_db, ber, mse_loss = evaluate_competition_faster([model,], test_loader,
                                                                          'QPSK', device)
plot_competition_figures(intrf_sig_names, all_sinr_db, mse_loss[0], mse_loss[1], ber[0], ber[1], 'QPSK')

