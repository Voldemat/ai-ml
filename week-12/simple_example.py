import numpy as np

def generate_dataset(
    random_generator: np.random.Generator,
    dataset_size: int
) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    inputs = 2 * random_generator.random(size=(dataset_size, 1))
    noise = random_generator.standard_normal(size=(dataset_size, 1)) * 0.4
    outputs = 4 + 3 * inputs + noise
    return (inputs, outputs)

