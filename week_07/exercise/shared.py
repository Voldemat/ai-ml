from typing import Callable, cast
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def create_normalize_dataset_function(
    reference_dataset: np.typing.NDArray[np.float64],
) -> Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]:
    d_min: np.typing.NDArray[np.float64] = reference_dataset.min(axis=0)
    d_max: np.typing.NDArray[np.float64] = reference_dataset.max(axis=0)
    d_range = d_max - d_min
    return lambda dataset: (dataset - d_min) / d_range


x_unnormalized, y = cast(
    tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.int64]],
    make_blobs(n_samples=300, centers=3, n_features=2, random_state=42),
)

x = create_normalize_dataset_function(x_unnormalized)(x_unnormalized)

x_train, x_test, y_train, y_test = cast(
    tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.int64],
        np.typing.NDArray[np.int64],
    ],
    cast(
        object,
        train_test_split(x, y, test_size=0.2, random_state=42, stratify=y),
    ),
)
