from collections.abc import Iterator
from typing import Any, Literal, TypedDict

from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class MoveDict(TypedDict, total=False):
    move: str
    time: int


class MovesDict(TypedDict, total=False):
    white: MoveDict
    black: MoveDict


class GameDict(TypedDict, total=False):
    ratings: tuple[int, int]
    moves: list[MovesDict]
    result: int
    total_time: int


class CustomDataset(Dataset[Tensor]):
    def __init__(self, data: Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]


class CustomDataLoader:
    def __init__(
        self, dataset: CustomDataset, batch_size: int = 1, **kwargs: Any
    ) -> None:
        self.dataset = dataset
        self._dataloader: DataLoader[Tensor] = DataLoader(
            dataset, batch_size=batch_size, **kwargs
        )

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self._dataloader)

    def __len__(self) -> int:
        return len(self.dataset)


class InputData(BaseModel):
    fen_number: str
    white_time: str
    black_time: str
    white_rating: str
    black_rating: str
    turn: str


class TrainingParams(TypedDict, total=False):
    epochs: int
    lr: float
    gamma: float
    scheduler: Literal["step", "exponential"]
    batch_size: int
    step_size: int
    optimizer: Literal["SGD", "Adam"]


class TuningParams(TrainingParams, total=False):
    n_hidden: int
    h1: int
    h2: int
    h3: int
