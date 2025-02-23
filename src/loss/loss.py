import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, noise, **batch):
        return {"loss": torch.nn.functional.mse_loss(predict, noise)}
