from typing import Any, cast
import numpy as np
from pandas import DataFrame
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from numpy_linear_regression import (
    compute_mean_squared_loss,
    predict,
    train_weights,
)


def main() -> None:
    housing = cast(Any, fetch_california_housing(as_frame=True))

    df: DataFrame = housing.frame.copy()

    target_col = "MedHouseVal"
    feature_cols = [c for c in df.columns if c != target_col]
    print("Dataset shape:", df.shape)
    print("Features:", feature_cols)
    inputs = df[feature_cols]
    outputs = df[target_col]

    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
        inputs, outputs, test_size=0.2, random_state=42
    )
    print("Train shape:", inputs_train.shape)
    print("Test shape:", inputs_test.shape)

    scaler = StandardScaler()
    inputs_train_scaled = scaler.fit_transform(inputs_train)
    inputs_test_scaled = scaler.transform(inputs_test)
    initial_weights = np.zeros(inputs_train_scaled.shape[1])
    weights = train_weights(
        inputs_train_scaled,
        outputs_train.values,
        initial_weights,
        max_iterations=8000,
        learning_rate=0.08,
        min_change=1e-10,
    )
    outputs_prediction = predict(inputs_test_scaled, weights)
    mean_squared_error = np.mean(
        (outputs_test.values - outputs_prediction) ** 2
    )
    root_mean_squared_error = np.sqrt(mean_squared_error)
    mean_absolute_error = np.mean(
        np.abs(outputs_test.values - outputs_prediction)
    )
    coefficient_of_determination = 1 - np.sum(
        (outputs_test - outputs_prediction) ** 2
    ) / np.sum((outputs_test - np.mean(outputs_test)) ** 2)


if __name__ == "__main__":
    main()
