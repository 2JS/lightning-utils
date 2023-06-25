from typing import Sequence, Tuple

import pytorch_lightning as pl
import torch


class DummyDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        *args: Sequence[Sequence[int]] | Sequence[int] | Sequence[torch.Tensor],
        range: Tuple[int, int] = (-1, 1),
    ):
        super().__init__()

        if isinstance(args[0], Sequence):
            self.sample_tensors = [torch.rand(*args) for args in args]
        elif isinstance(args[0], int):
            self.sample_tensors = [torch.rand(*args)]
        elif isinstance(args[0], torch.Tensor):
            self.sample_tensors = args
        else:
            raise TypeError(
                f"expected Sequence[Sequence[int]] | Sequence[int] | Sequence[torch.Tensor], got {args}"
            )

        self.range = range

    def __iter__(self):
        # if not isinstance(self.shapes[0], tuple):
        #     return torch.rand(*self.shapes, self.range[0], self.range[1])
        return iter(range(10))


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = DummyDataset((1, 2, 3), (4, 5, 6), (7, 8, 9))
    dataloader = DataLoader(dataset, batch_size=2)
    print(next(dataloader))
    print(next(dataloader))
