import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
regr.predict(X_test[:2])
regr.score(X_test, y_test)

NN4 = MLPRegressor(hidden_layer_sizes=[1000,200], activation='relu').fit(X_train, y_train.ravel())
grid = np.linspace(-6,6,1000)
preds = NN4.predict(grid.reshape(-1,1))

plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')

plt.plot(grid, preds, label='NN prediction', c='r')
plt.legend()

# 2.1 Алгоритм K-ближайших соседей.
KNN = KNeighborsRegressor().fit(X_train, y_train)
grid = np.linspace(-6,6,1000)
preds_KNN = KNN.predict(grid.reshape(-1,1))

plt.figure(figsize=(8,4))
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')

plt.plot(grid, preds_KNN, label='KNeighborsRegressor prediction', c='r')
plt.legend()

# 2.2 Алгоритм «деревья решения».
tree = DecisionTreeRegressor().fit(X_train, y_train)
preds_tree = tree.predict(grid.reshape(-1,1))

plt.figure(figsize=(8,4))
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')

plt.plot(grid, preds_tree, label='DecisionTreeRegressor prediction', c='r')
plt.legend()

# 2.3 Алгоритм «градиентный бустинг»
grad_boost = GradientBoostingRegressor().fit(X_train, y_train.ravel())
preds_grad_boost = grad_boost.predict(grid.reshape(-1,1))

plt.figure(figsize=(8,4))
plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, label='test')

plt.plot(grid, preds_grad_boost, label='GradientBoostingRegressor prediction', c='r')
plt.legend()

# 2.4 Вывод результатов тренировочных данных по всем алгоритмам 
# ML и результатам нейросети.
plt.figure(figsize=(10,10))
cos = np.cos(grid)

plt.plot(grid, preds_KNN, label='KNeighborsRegressor prediction')
plt.plot(grid, preds_tree, label='DecisionTreeRegressor prediction')
plt.plot(grid, preds_grad_boost, label='GradientBoostingRegressor prediction')
plt.plot(grid, preds, label='NN prediction')
plt.plot(grid, cos, label='Эталонная функция')
plt.legend()
plt.show()