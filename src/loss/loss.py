import torch
from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predict, target_img_latent, **batch):
        return {"loss": torch.nn.functional.mse_loss(predict, target_img_latent)}
