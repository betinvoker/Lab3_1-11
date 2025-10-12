import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

california = fetch_california_housing()
X, y = california.data, california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

normalizer1 = tf.keras.layers.Normalization(axis=-1)
normalizer2 = tf.keras.layers.Normalization(axis=-1)

normalizer1.adapt(np.array(X_train))
normalizer2.adapt(np.array(X_test))

X_train_new = normalizer1(X_train).numpy()
X_test_new = normalizer2(X_test).numpy()

model4 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(1) # регрессия - один выход
])
model4.compile(optimizer='adam', loss='mse', metrics=['mae'])

history4=model4.fit(X_train_new, y_train, epochs=50, 
                    batch_size=32, verbose=0)
model4.summary()

keras.utils.plot_model(model4, show_shapes=True)

mse, mae = model4.evaluate(X_test, y_test, verbose=0)
print('Средняя абсолютная ошибка (тысяча долларов):', mae)

pred = model4.predict(X_test)

print(f'Предсказанная стоимость: {pred[1][0]}, правильная стоимость: {y_test[1]}')
print(f'Предсказанная стоимость: {pred[50][0]}, правильная стоимость: {y_test[50]}')
print(f'Предсказанная стоимость: {pred[100][0]}, правильная стоимость: {y_test[100]}')

plt.plot(history4.history['loss'], label='loss 4')
plt.xlabel('Эпоха обучения')
plt.legend()
plt.show()