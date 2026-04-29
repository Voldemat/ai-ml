from typing import cast
import numpy as np
from .shared import x_train, x_test, y_train, y_test


def euclidian_distance(
    x_test_dataset: np.typing.NDArray[np.float64],
    x_train_dataset: np.typing.NDArray[np.float64],
) -> np.typing.NDArray[np.float64]:
    diff = x_test_dataset[:, np.newaxis, :] - x_train_dataset[np.newaxis, :, :]
    return np.sqrt(cast(np.typing.NDArray[np.float64], np.sum(diff**2, axis=2)))


def predict(
    x_test_dataset: np.typing.NDArray[np.float64],
    x_train_dataset: np.typing.NDArray[np.float64],
    y_train_dataset: np.typing.NDArray[np.int64],
    k: int,
) -> np.typing.NDArray[np.int64]:
    distance = np.argsort(
        euclidian_distance(x_test_dataset, x_train_dataset), axis=1
    )
    k_nearest_idx = distance[:, :k]
    k_nearest_labels = y_train_dataset[k_nearest_idx]
    return np.array(
        [
            np.bincount(row).argmax()
            for row in map(
                lambda r: cast(  # pyright: ignore[reportAny]
                    np.typing.NDArray[np.int64], r
                ),
                k_nearest_labels,
            )
        ]
    )


def main() -> None:
    y_predicted = predict(x_test, x_train, y_train, 1)
    is_correct: np.typing.NDArray[np.bool] = y_predicted == y_test
    accuracy = np.mean(is_correct)
    print(f"Accuracy: {accuracy:.4f}")
    

