import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

X_train = np.random.uniform(-6,6, size=(500,1))
y_train = np.cos(X_train)

X_test = np.random.uniform(-6,6, size=(100,1))
y_test = np.cos(X_test)

plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')
plt.legend()
plt.show()