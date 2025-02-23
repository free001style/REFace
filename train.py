import warnings
import os

import hydra
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.model_utils import instantiate_model

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="src/configs", config_name="reface")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)
    from accelerate import DistributedDataParallelKwargs
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    project_config = OmegaConf.to_container(config)
    if accelerator.is_main_process:
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
    else:
        logger = None
        writer = None

    device = accelerator.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate_model(config.model, device).to(device)
    if accelerator.is_main_process:
        logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = (
        list(model.model.parameters())
        + list(model.landmark_proj_out.parameters())
        + list(model.last_proj.parameters())
    )
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    epoch_len = config.trainer.get("epoch_len")
    
    model, optimizer, lr_scheduler= accelerator.prepare(model, optimizer, lr_scheduler)

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        ema=None,
        accelerator=accelerator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
