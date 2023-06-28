from typing import Sequence

import torch


class DummyTensorDataset(torch.utils.data.IterableDataset):
    def __init__(self, *example_tensors):
        super().__init__()
        self.generator = lambda: tuple(torch.randn_like(t) for t in example_tensors)

    def __iter__(self):
        while True:
            yield self.generator()


class DummyTensorDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *example_tensors, **kwargs):
        super().__init__(
            DummyTensorDataset(*example_tensors),
            **kwargs,
        )


if __name__ == "__main__":
    dataset = DummyTensorDataset(torch.zeros(1, 2, 3), torch.zeros(4, 5, 6))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    iterator = iter(dataloader)

    print(next(iterator)[0].shape)
    print(next(iterator)[1].shape)
