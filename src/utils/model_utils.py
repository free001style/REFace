from collections import OrderedDict

import torch
from hydra.utils import instantiate


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def instantiate_model(config, device):
    unet = instantiate(config.unet)
    unet.load_state_dict(
        torch.load(config.sd_ckpt, map_location="cpu"),
        strict=False,
    )
    vae = instantiate(config.vae)
    model = instantiate(config.reface, unet=unet, vae=vae, device=device)
    return model
