from typing import Callable, cast
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)


def create_normalize_dataset_function(
    reference_dataset: np.typing.NDArray[np.float64],
) -> Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]:
    d_min: np.typing.NDArray[np.float64] = reference_dataset.min(axis=0)
    d_max: np.typing.NDArray[np.float64] = reference_dataset.max(axis=0)
    d_range = d_max - d_min
    return lambda dataset: (dataset - d_min) / d_range


n_per_class = 25
x_0_unnormalized = np.random.randn(n_per_class, 2) + np.array([2, 2])
normalize_dataset = create_normalize_dataset_function(x_0_unnormalized)
x_0 = normalize_dataset(x_0_unnormalized)
y_0: np.typing.NDArray[np.int64] = np.zeros(n_per_class, dtype=int)

x_1 = normalize_dataset(np.random.randn(n_per_class, 2) + np.array([3.5, 3.5]))
y_1: np.typing.NDArray[np.int64] = np.ones(n_per_class, dtype=int)

x = np.vstack([x_0, x_1])
y = np.hstack([y_0, y_1])

x_train, x_test, y_train, y_test = cast(
    tuple[
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.float64],
        np.typing.NDArray[np.int64],
        np.typing.NDArray[np.int64],
    ],
    cast(object, train_test_split(x, y, test_size=0.2, random_state=42)),
)

print(f"Training samples per class: {n_per_class}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")
k = 5
print(f"K: {k}")
