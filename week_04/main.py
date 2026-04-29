from typing import Any, cast
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
hours = np.array(
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    dtype=np.float64
)
print(f"{hours=}")

y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float64).reshape(-1, 1)
print(f"{y=}")
x = np.column_stack([np.ones_like(hours), hours])

def sigmoid(z: Any) -> Any:
    clipped_z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped_z))

def forward(x, w):
    return sigmoid(x @ w)

def binary_cross_entropy(y, p_hat, eps=1e-15):
    p_hat = np.clip(p_hat, eps, 1 - eps)
    n = y.shape[0]
    return -np.mean(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

def gradient(x, y, p_hat):
    n = x.shape[0]
    error = p_hat - y
    return (x.T @ error) / n


n_features = x.shape[1]
learning_rate = 0.5
epochs = 10000

w = np.random.randn(n_features, 1) * 0.1
print("Initial weights: ", w)
loss_history = []

for epoch in range(epochs):
    p_hat = forward(x, w)
    loss = binary_cross_entropy(y, p_hat)
    loss_history.append(loss)
    grad = gradient(x, y, p_hat)
    w = w - learning_rate * grad

print(f"Final weights: intercept - {w[0, 0]}, slope - {w[1, 0]}")

boundary_hours = -w[0, 0] / w[1, 0]
hours_curve = np.linspace(0, 6, 200)
x_curve = np.column_stack([np.ones_like(hours_curve), hours_curve])
p_curve = forward(x_curve, w)

def predict_class(x, w, threshold = 0.5):
    p_hat = forward(x, w)
    return (p_hat >= threshold).astype(np.float64)

y_pred = predict_class(x, w)

print("Hours | True | Predicted | P(Pass)")
print("-" * 40)

p_hat = forward(x, w)
for i in range(len(hours)):
    print(f"{hours[i]:5.1f} | {int(y[i,0]):4d} | {int(y_pred[i,0]):9d} | {p_hat[i,0]:.3f}")

accuracy = np.mean(y_pred == y)
print(f"\nTraining accuracy: {accuracy*100:.0f}%")

def predict_hours(hours_new, w):
    hours_new = np.atleast_1d(hours_new)
    x_new = np.column_stack([np.ones_like(hours_new), hours_new])
    p = forward(x_new, w)
    y_pred = (p >= 0.5).astype(np.float64)
    return p, y_pred

for h in [2.2, 4.2]:
    p, pred = predict_hours(h, w)
    outcome = "Pass" if pred[0, 0] == 1 else "Fail"
    print(f"{h} hours → P(Pass) = {p[0,0]:.3f} → Predict: {outcome}")
