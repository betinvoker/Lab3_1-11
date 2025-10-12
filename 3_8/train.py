import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

california = fetch_california_housing()
X, y = california.data, california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = pd.DataFrame(data=X_train)
print(df.describe().transpose()[['mean', 'std']])

df_test = pd.DataFrame(data=X_test)
df_test.describe().transpose()[['mean', 'std']]

# Среднее значение
mean = X_train.mean(axis=0)
# Стандартное отклонение
std = X_train.std(axis=0)
X_train -= mean
X_train /= std
X_test -= mean
X_test /= std

df_train = pd.DataFrame(data=X_train)
print(df_train.describe().transpose()[['mean', 'std']])

df_test = pd.DataFrame(data=X_test)
print(df_test.describe().transpose()[['mean', 'std']])

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(1) # регрессия - один выход
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history1 = model.fit(X_train, y_train, epochs=50, 
                     batch_size=32, verbose=2)
model.summary()

keras.utils.plot_model(model, show_shapes=True)

mse, mae = model.evaluate(X_test, y_test, verbose=0)
print('Средняя абсолютная ошибка (тысяча долларов):', mae)

pred = model.predict(X_test)

print(f'Предсказанная стоимость: {pred[1][0]}, правильная стоимость: {y_test[1]}')
print(f'Предсказанная стоимость: {pred[50][0]}, правильная стоимость: {y_test[50]}')
print(f'Предсказанная стоимость: {pred[100][0]}, правильная стоимость: {y_test[100]}')

plt.plot(history1.history['loss'], label='loss 1')
plt.xlabel('Эпоха обучения')
plt.legend()
# plt.show()

model2 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1) # регрессия - один выход
])
model2.compile(optimizer='adam', loss='mse', metrics=['mae'])

history2 = model2.fit(X_train, y_train, epochs=100, 
                     batch_size=16, verbose=1)
model2.summary()

keras.utils.plot_model(model2, show_shapes=True)

mse, mae = model2.evaluate(X_test, y_test, verbose=0)
print('Средняя абсолютная ошибка (тысяча долларов):', mae)

pred2 = model2.predict(X_test)

print(f'Предсказанная стоимость: {pred2[1][0]}, правильная стоимость: {y_test[1]}')
print(f'Предсказанная стоимость: {pred2[50][0]}, правильная стоимость: {y_test[50]}')
print(f'Предсказанная стоимость: {pred2[100][0]}, правильная стоимость: {y_test[100]}')

plt.plot(history2.history['loss'], label='loss 2')
plt.xlabel('Эпоха обучения')
plt.legend()

model3 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])
model3.compile(optimizer='adam', loss='mse', metrics=['mae'])

history3 = model3.fit(X_train, y_train, epochs=80, 
                      batch_size=8, verbose=1)
model3.summary()

keras.utils.plot_model(model3, show_shapes=True)

mse3, mae3 = model3.evaluate(X_test, y_test, verbose=0)
print(f"\nСредняя абсолютная ошибка (тысяч долларов) для model3: {mae3:.4f}")

pred3 = model3.predict(X_test)

print(f'Предсказанная стоимость: {pred3[1][0]},правильная стоимость: {y_test[1]}')
print(f'Предсказанная стоимость: {pred3[50][0]},правильная стоимость: {y_test[50]}')
print(f'Предсказанная стоимость: {pred3 [100][0]},правильная стоимость: {y_test[100]}')

plt.plot(history3.history['loss'], label='loss 3')
plt.xlabel('Эпоха обучения')
plt.legend()
plt.show()