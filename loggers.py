from functools import wraps
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from pytorch_lightning.trainer.states import RunningStage


def _log(
    pl_module: pl.LightningModule,
    result: torch.Tensor | dict,
    prefix: Optional[str] = None,
    rank_zero_only: bool = True,
    **log_kwargs,
):
    if isinstance(result, torch.Tensor):
        pl_module.log(
            f"{prefix}loss",
            result,
            rank_zero_only=rank_zero_only,
            **log_kwargs,
        )
    elif isinstance(result, dict):
        _result = {f"{prefix}{k}": v for k, v in result.items() if k[0] != "_"}
        pl_module.log_dict(
            _result,
            rank_zero_only=rank_zero_only,
            **log_kwargs,
        )
    else:
        raise TypeError(f"result type {type(result)} is not supported for logging")


def log(
    func: callable = None,
    prefix: Optional[str] = None,
    rank_zero_only: bool = True,
    **log_kwargs,
):
    if func is None:
        return lambda func: log(func, prefix)

    @wraps(func)
    def wrapper(self: pl.LightningModule, *args, **kwargs):
        assert isinstance(
            self, pl.LightningModule
        ), "log decorator can only be used on LightningModule methods"

        result = func(self, *args, **kwargs)

        stage = self.trainer.state.stage

        if prefix is None:
            if stage == RunningStage.TRAINING:
                prefix = "train_"
            elif stage == RunningStage.VALIDATING:
                prefix = "valid_"
            elif stage == RunningStage.TESTING:
                prefix = "test_"
            elif stage == RunningStage.SANITY_CHECKING:
                prefix = "sanity_"
            elif stage == RunningStage.PREDICTING:
                prefix = "predict_"
            else:
                prefix = "unknown_"

        _log(self, result, prefix, rank_zero_only, **log_kwargs)

        return result

    return wrapper


class OutputLogger(pl.Callback):
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        _log(pl_module, outputs, prefix="train_")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        _log(pl_module, outputs, prefix="valid_")
