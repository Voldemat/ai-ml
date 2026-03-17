from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from .shared import k, x_test, x_train, y_train, y_test

def main() -> None:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights='uniform',
        metric='euclidean',
        algorithm='auto'
    )
    _ = knn.fit(x_train, y_train)

    y_pred_sklearn = knn.predict(x_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"k: {k} - Accuracy: {acc_sklearn:.4f}")
    print("True:", y_test[:10])
    print("Pred:", y_pred_sklearn[:10])
