import torch
from einops import rearrange

from src.metrics.tracker import MetricTracker
from src.model.ldm.utils import denormalize
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        with self.accelerator.accumulate(self.model):
            batch = self.move_batch_to_device(batch)
            batch = self.transform_batch(batch)  # transform batch on device -- faster

            metric_funcs = self.metrics["inference"]
            if self.is_train:
                metric_funcs = self.metrics["train"]
                self.optimizer.zero_grad()

            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

            if self.is_train:
                self.accelerator.backward(batch["loss"])
                self._clip_grad_norm()
                self.optimizer.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train", N_row=7):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        batch = self.move_batch_to_device(batch)
        size = min(N_row, batch["source_img"].shape[0])
        for key in batch:
            if key != "loss" and batch[key] is not None:
                batch[key] = batch[key][:size]
        self.model.eval()
        if torch.cuda.device_count() == 1:
            samples = self.model.sample(prepare=mode != "train", **batch)
        else:
            samples = self.model.module.sample(prepare=mode != "train", **batch)
        if mode == "train":
            if batch["corrupt_img"] is not None:
                img = torch.cat(
                    [
                        batch["source_img"],
                        batch["target_img"],
                        batch["corrupt_img"],
                        batch["inpaint_img"],
                        torch.cat([batch["mask"]] * 3, dim=1),
                        samples,
                    ],
                    dim=-1,
                )
            else:
                img = torch.cat(
                [batch["source_img"], batch["target_img"], batch["inpaint_img"], torch.cat([batch["mask"]] * 3, dim=1), samples], dim=-1
            )
                
        else:
            img = torch.cat(
                    [
                        batch["source_img"], # only_source_img
                        batch["target_img"],
                        samples,
                    ],
                    dim=-1,
                )
            
        img = rearrange(img, "b c h w -> c (b h) w").clip_(-1, 1)
        img = denormalize(img)
        self.writer.add_image(f"{mode}_img", img)
        self.model.train()
