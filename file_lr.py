from torch.optim.lr_scheduler import LRScheduler
from pytorch_lightning.callbacks import Callback


class FileLR(LRScheduler):
    def __init__(self, optimizer, path="lr", *args, **kwargs):
        super(FileLR, self).__init__(optimizer, *args, **kwargs)
        self.path = path

    def get_lr(self):
        with open(self.path, "r") as f:
            lr = float(f.read())
        return [lr] * len(self.base_lrs)


class FileLRCallback(Callback):
    def __init__(self, path="lr"):
        self.path = path

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        for config in trainer.lr_scheduler_configs:
            optimizer = config.scheduler.optimizer
            config.scheduler = FileLR(optimizer, self.path)
            config.interval = "step"
            config.frequency = 1
            config.reduce_on_plateau = False


if __name__ == "__main__":
    import torch
    import pytorch_lightning as pl

    class Module(pl.LightningModule):
        def forward(self, x):
            return x

    trainer = pl.Trainer(
        callbacks=[FileLRCallback()],
        max_epochs=10,
    )
