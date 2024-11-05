import random
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import yaml

from .custom_types import Config

T = TypeVar("T", bound=np.generic)
R = TypeVar("R", bound=np.generic)
TrainTestIds = tuple[list[int], list[int]]


def parse_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        config: Config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def train_test_split(
    x: npt.NDArray[T],
    y: npt.NDArray[R],
    test_ratio: float,
    random_seed: int | None = None,
) -> tuple[npt.NDArray[T], npt.NDArray[R], npt.NDArray[T], npt.NDArray[R]]:
    id_range = range(len(x))
    train_ids, test_ids = split_train_test_id(id_range, test_ratio, random_seed)

    x_train = x[train_ids]
    y_train = y[train_ids]
    x_test = x[test_ids]
    y_test = y[test_ids]

    return x_train, y_train, x_test, y_test


def split_train_test_id(
    id_range: range,
    test_ratio: float,
    random_seed: int | None = None,
) -> TrainTestIds:
    random.seed(random_seed)

    n_tests = int(len(id_range) * test_ratio)

    test_ids = list(random.sample(id_range, n_tests))
    test_id_set = set(test_ids)
    train_ids = [i for i in id_range if i not in test_id_set]

    return train_ids, test_ids
