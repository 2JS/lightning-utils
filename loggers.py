from functools import wraps
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer.states import RunningStage


def log(
    func: callable = None,
    prefix: Optional[str] = None,
    rank_zero_only: bool = True,
    **kwargs,
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
                prefix = "train"
            elif stage == RunningStage.VALIDATING:
                prefix = "valid"
            elif stage == RunningStage.TESTING:
                prefix = "test"
            elif stage == RunningStage.SANITY_CHECKING:
                prefix = "sanity"
            elif stage == RunningStage.PREDICTING:
                prefix = "predict"
            else:
                prefix = "unknown"

        if isinstance(result, torch.Tensor):
            self.log(
                f"{prefix}_loss",
                result,
                rank_zero_only=rank_zero_only,
                **kwargs,
            )
        elif isinstance(result, dict):
            _result = {f"{prefix}_{k}": v for k, v in result.items()}
            self.log_dict(
                _result,
                rank_zero_only=rank_zero_only,
                **kwargs,
            )
        else:
            print(f"result type {type(result)} is not supported for logging")

        return result

    return wrapper
