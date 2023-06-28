from torch.optim.lr_scheduler import LRScheduler
from pytorch_lightning.callbacks import Callback


class FileLR(LRScheduler):
    def __init__(self, optimizer, path="lr", *args, **kwargs):
        self.path = path
        super().__init__(optimizer, *args, **kwargs)

    def get_lr(self):
        with open(self.path, "r") as f:
            lr = float(f.read())
        return [lr] * len(self.base_lrs)


class FileLRCallback(Callback):
    def __init__(self, path='lr'):
        self.path = path

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        for config in trainer.lr_scheduler_configs:
            optimizer = config.scheduler.optimizer
            config.scheduler = FileLR(optimizer, self.path)
            config.interval = 'step'
            config.frequency = 1
            config.reduce_on_plateau = False


if __name__ == "__main__":
    import torch
    import pytorch_lightning as pl
    from dummy import DummyTensorDataModule

    class Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

        def training_step(self, batch, batch_idx):
            (x,) = batch
            loss = x.sum() + self.param
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
            scheduler = FileLR(optimizer, path="lr")
            return [optimizer], [scheduler]

        def on_before_optimizer_step(self, optimizer):
            print(optimizer.param_groups[0]["lr"])

    trainer = pl.Trainer(
        callbacks=[FileLRCallback()],
        max_steps=10,
    )

    trainer.fit(
        model=Module(),
        datamodule=DummyTensorDataModule(torch.zeros(3, 4, 5), batch_size=2),
    )
