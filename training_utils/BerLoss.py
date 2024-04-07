from torch import nn
from comm_utils import split_to_complex_batch
import torch


class BerLoss(nn.Module):
    def __init__(self, demodulator, device):
        super(BerLoss, self).__init__()
        self.demodulator = demodulator
        self.device = device

    def forward(self, sig_pred, msg_bits):
        pred_tf = split_to_complex_batch(sig_pred.detach().cpu())
        bits, _ = self.demodulator(pred_tf)
        bits = torch.tensor(bits, dtype=torch.float32).to(self.device)
        ber = torch.mean(torch.abs(bits - msg_bits))
        ber.requires_grad_(True)
        return ber

