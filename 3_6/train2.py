import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X_train = np.random.uniform(-6,6, size=(500,1))
y_train = np.cos(X_train)

X_test = np.random.uniform(-6,6, size=(100,1))
y_test = np.cos(X_test)

# X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
# regr.predict(X_test[:2])
# regr.score(X_test, y_test)

NN1 = MLPRegressor(random_state=1, max_iter=1000).fit(X_train, y_train.ravel())
grid = np.linspace(-6,6,1000)
preds = NN1.predict(grid.reshape(-1,1))

plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')

plt.plot(grid, preds, label='NN prediction', c='r')
plt.legend()
plt.show()