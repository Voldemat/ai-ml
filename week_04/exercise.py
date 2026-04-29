import numpy as np

np.random.seed(42)

hours_studied = np.array(
    [1.2, 1.5, 2.0, 2.2, 2.8, 3.0, 3.2, 3.5, 4.0, 4.2, 4.5, 4.8, 5.0, 5.2, 5.5],
    dtype=np.float64,
)
hours_slept = np.array(
    [5.0, 6.0, 5.5, 6.5, 7.0, 6.0, 7.5, 7.0, 8.0, 7.0, 8.0, 7.5, 8.0, 6.5, 8.0],
    dtype=np.float64,
)
attendance = np.array(
    [
        0.6,
        0.7,
        0.5,
        0.8,
        0.6,
        0.9,
        0.7,
        0.85,
        0.9,
        0.95,
        0.8,
        0.9,
        1.0,
        0.75,
        0.95,
    ],
    dtype=np.float64,
)

passed = np.array(
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64
).reshape(-1, 1)

features = np.column_stack(
    [np.ones_like(hours_studied), hours_studied, hours_slept, attendance]
)

def sigmoid(data):
    return 1.0 / (1.0 + np.exp(-np.clip(data, -500, 500)))

def forward(data, weights):
    return sigmoid(data @ weights)

def binary_loss_entropy(result, predicted, eps=1e-15):
    predicted = np.clip(predicted, eps, 1 - eps)
    return -np.mean(result * np.log(predicted) + (1-result) * np.log(1 - predicted))

def gradient(data, result, predicted):
    n = data.shape[0]
    error = predicted - result
    return data.T @ error


n_features = features.shape[1]
learning_rate = 0.5
epochs = 1000
loss_history = []

weights = np.random.randn(n_features, 1) * 0.1

for epoch in range(epochs):
    prediction = forward(features, weights)
    error = binary_loss_entropy(passed, prediction)
    grad = gradient(features, passed, prediction)
    weights = weights - learning_rate * grad
    loss_history.append(error)

print(f"Final weights: {weights}")
print(f"Last error: {loss_history[len(loss_history) - 1]}")

prediction = (forward(features, weights) >= 0.5).astype(np.float64)
accuracy = np.mean(prediction == passed)
print(f"Accuracy: ", accuracy)

new_student = np.array([[1, 3.8, 7, 0.88]])
print(f"New student passed?: ", forward(new_student, weights) >= 0.5)
