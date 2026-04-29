import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from dataset import x, x_train, x_test, y_train, y_test

lin = LinearRegression()
lin.fit(x_train, y_train)

y_train_pred = lin.predict(x_train)
y_test_pred = lin.predict(x_test)

train_mse_lin = mean_squared_error(y_train, y_train_pred)
test_mse_lin = mean_squared_error(y_test, y_test_pred)

xx = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
yy_true = np.sin(xx).ravel()
yy_hat = lin.predict(xx)

fix, axes = plt.subplots(1, 2, figsize=(12, 4.5))
