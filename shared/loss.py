import numpy as np

def compute_mean_squared_loss(
    test_dataset: np.typing.NDArray[np.float64],
    predicted_dataset: np.typing.NDArray[np.float64],
) -> float:
    return float(np.mean((test_dataset - predicted_dataset) ** 2))

