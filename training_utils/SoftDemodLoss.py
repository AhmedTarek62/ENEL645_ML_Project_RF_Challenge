from torch import nn
from comm_utils import split_to_complex_batch
import torch
import numpy as np


class SoftDemodLoss(nn.Module):
    def __init__(self, demodulator, device, resource_grid=None):
        super(SoftDemodLoss, self).__init__()
        self.demodulator = demodulator
        self.resource_grid = resource_grid
        self.device = device
        self.criterion = nn.MSELoss()

    def forward(self, sig_pred, sig_target):
        pred_tf = split_to_complex_batch(sig_pred.detach().cpu())
        target_tf = split_to_complex_batch(sig_target.detach().cpu())
        if self.resource_grid is None:
            _, soft_pred = self.demodulator(pred_tf)
            _, soft_target = self.demodulator(target_tf)
        else:
            _, soft_pred = self.demodulator(pred_tf, self.resource_grid)
            _, soft_target = self.demodulator(target_tf, self.resource_grid)
        soft_pred_real = torch.tensor(np.real(soft_pred), dtype=torch.float32).to(self.device)
        soft_target_real = torch.tensor(np.real(soft_target), dtype=torch.float32).to(self.device)
        soft_pred_imag = torch.tensor(np.imag(soft_pred), dtype=torch.float32).to(self.device)
        soft_target_imag = torch.tensor(np.imag(soft_target), dtype=torch.float32).to(self.device)
        soft_loss = self.criterion(soft_target_real, soft_pred_real) + self.criterion(soft_target_imag, soft_pred_imag)
        mse_loss = self.criterion(sig_pred, sig_target)
        loss = soft_loss + mse_loss
        loss.requires_grad_(True)
        return loss

