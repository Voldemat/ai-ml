import numpy as np
from sklearn.linear_model import LogisticRegression

hours = np.array(
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], dtype=np.float64
)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)

X = hours.reshape(-1, 1)

model = LogisticRegression(fit_intercept=True, random_state=42)
model.fit(X, y)

y_pred = model.predict(X)

y_proba = model.predict_proba(X)[:, 1]
print("Predictions:", y_pred)
print("P(pass):    ", np.round(y_proba, 3))
print("Training accuracy:", (y_pred == y).mean())

print("Intercept (bias):", model.intercept_[0])
print("Coefficient (hours):", model.coef_[0, 0])
boundary = -model.intercept_[0] / model.coef_[0, 0]
print("Decision boundary (hours):", round(boundary, 2))

X_new = np.array([[2.2], [4.2]])
pred_new = model.predict(X_new)
proba_new = model.predict_proba(X_new)[:, 1]
for hrs, p, prob in zip(X_new.ravel(), pred_new, proba_new):
    print(f"{hrs} hours → P(pass) = {prob:.3f} → {'Pass' if p == 1 else 'Fail'}")
