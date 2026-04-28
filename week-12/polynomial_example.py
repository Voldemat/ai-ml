import numpy as np


def generate_dataset(
    random_generator: np.random.Generator, dataset_size: int
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    inputs = random_generator.uniform(-3, 3, size=(dataset_size, 1))
    outputs = (
        1.5
        - 2.0 * inputs
        + 0.9 * (inputs**2)
        + 0.2 * (inputs**3)
        + random_generator.standard_normal(size=(dataset_size, 1)) * 1.2
    )
    return (inputs, outputs)
