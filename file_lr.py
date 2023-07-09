from pytorch_lightning.callbacks import Callback
from torch.distributed import broadcast_object_list
from torch.optim.lr_scheduler import LRScheduler


class FileLR(LRScheduler):
    def __init__(self, optimizer, path="lr", *args, **kwargs):
        self.path = path
        super().__init__(optimizer, *args, **kwargs)

    def get_lr(self):
        with open(self.path, "r") as f:
            lr = float(f.read())
        return [lr] * len(self.base_lrs)


class FileLRCallback(Callback):
    def __init__(self, path="lr"):
        super().__init__()
        self.path = path

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str,
    ):
        with open(self.path, "r") as f:
            lr = float(f.read())

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: "torch.optim.Optimizer",
    ):
        lr = 0

        if trainer.is_global_zero:
            with open(self.path, "r") as f:
                lr = float(f.read())

        if trainer.world_size > 1:
            broadcast_object_list([lr], src=0)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == "__main__":
    import pytorch_lightning as pl
    import torch

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
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 1e-2)
            return [optimizer], [scheduler]

        def on_before_optimizer_step(self, optimizer):
            print(optimizer.param_groups[0]["lr"])

    trainer = pl.Trainer(
        callbacks=[FileLRCallback()],
        max_epochs=10,
        limit_train_batches=1,
    )

    trainer.fit(
        model=Module(),
        datamodule=DummyTensorDataModule(torch.zeros(3, 4, 5), batch_size=2),
    )
