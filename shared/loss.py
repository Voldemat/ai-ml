import numpy as np


def compute_mean_squared_loss(
    test_dataset: np.typing.NDArray[np.float64],
    predicted_dataset: np.typing.NDArray[np.float64],
) -> float:
    return float(np.mean((test_dataset - predicted_dataset) ** 2))


def compute_binary_cross_entropy(
    test_dataset: np.typing.NDArray[np.float64],
    predicted_dataset: np.typing.NDArray[np.float64],
    eps: float = 1e-15,
) -> float:
    clipped_predicted_dataset = np.clip(predicted_dataset, eps, 1 - eps)
    return float(
        -np.mean(
            test_dataset * np.log(clipped_predicted_dataset)
            + (1 - test_dataset) * np.log(1 - clipped_predicted_dataset)
        )
    )
