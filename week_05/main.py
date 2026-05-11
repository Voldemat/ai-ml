from dataclasses import dataclass
import functools
import numpy as np

from typing import Any


def compute_gene_impurity(values: np.typing.NDArray[np.floating[Any]]) -> float:
    probabilities = np.bincount(values) / len(values)
    return 1 - np.sum(probabilities**2)


def number_to_age(v: int) -> str:
    match v:
        case 1:
            return "Young"
        case 2:
            return "Mid"
        case 3:
            return "Senior"
        case _:
            raise NotImplementedError()


def number_to_income(v: int) -> str:
    match v:
        case 1:
            return "High"
        case 0:
            return "Low"
        case _:
            raise NotImplementedError()


@dataclass
class ColumnState:
    index: int
    weighted_gene_impurity: float


def find_best_split(
    inputs: np.typing.NDArray[Any], outputs: np.typing.NDArray[Any]
) -> int:
    def map_function(column_index: int) -> ColumnState:
        values = inputs[:, column_index]
        unique_values = np.unique(values)

        def reduce_function(
            weighted_gene_impurity: float, unique_value: float
        ) -> float:
            local_indexes = np.where(values == unique_value)[0]
            local_labels = outputs[local_indexes]
            local_gene_impurity = compute_gene_impurity(local_labels)
            local_proportion = len(local_indexes) / len(values)
            return (
                weighted_gene_impurity + local_gene_impurity * local_proportion
            )

        weighted_gene_impurity = functools.reduce(
            reduce_function, unique_values, 0.0
        )
        return ColumnState(
            index=column_index, weighted_gene_impurity=weighted_gene_impurity
        )

    return functools.reduce(
        lambda state, current: (
            current
            if state.weighted_gene_impurity > current.weighted_gene_impurity
            or state.weighted_gene_impurity < 0
            else state
        ),
        map(map_function, range(0, inputs.shape[1])),
        ColumnState(index=-1, weighted_gene_impurity=-1.0),
    ).index


if __name__ == "__main__":
    dataset = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [2, 1, 1],
            [3, 0, 1],
            [3, 1, 1],
            [2, 0, 0],
            [1, 1, 1],
            [3, 0, 1],
        ]
    )
    inputs = dataset[:, 0:2]
    outputs = dataset[:, -1]
    print(find_best_split(inputs, outputs))
