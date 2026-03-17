from typing import cast
import numpy as np
from .shared import x_test, x_train, y_train, y_test, k


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
    k_coef: int,
) -> np.typing.NDArray[np.int64]:
    dists = euclidian_distance(x_test_dataset, x_train_dataset)
    k_nearest_idx = np.argsort(dists, axis=1)[:, :k_coef]
    neighbor_labels = y_train_dataset[k_nearest_idx]
    return np.array(
        [
            np.bincount(row).argmax()
            for row in map(
                lambda r: cast(  # pyright: ignore[reportAny]
                    np.typing.NDArray[np.int64], r
                ),
                neighbor_labels,
            )
        ]
    )


def main() -> None:
    y_pred_numpy = predict(x_test, x_train, y_train, k)
    acc_numpy = np.mean(
        cast(np.typing.NDArray[np.bool], y_pred_numpy == y_test)
    )
    print(f"Accuracy: {acc_numpy:.4f}")
    print("True:", y_test[:10])
    print("Pred:", y_pred_numpy[:10])
