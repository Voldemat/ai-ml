import numpy as np

from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(scale=0.2, size=X.shape[0])

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sort_idx = np.argsort(x_test.ravel())
x_test_sorted = x_test[sort_idx]
y_test_sorted = y_test[sort_idx]
