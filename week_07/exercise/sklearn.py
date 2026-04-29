from typing import cast
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from .shared import x_test, x_train, y_train, y_test
import matplotlib.pyplot as plt


def calc_accuracy_for_k(k: int) -> float:
    knn = KNeighborsClassifier(
        n_neighbors=k, weights="uniform", metric="euclidean", algorithm="auto"
    )
    _ = knn.fit(x_train, y_train)

    y_predicted = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    return float(accuracy)


def main() -> None:
    accuracy = calc_accuracy_for_k(5)
    print(f"Accuracy: {accuracy:.4f} for k = 5")
    k_range = range(1, cast(int, x_train.shape[0]))
    accuracy_list = list(map(calc_accuracy_for_k, k_range))
    plt.plot(k_range, accuracy_list)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()
