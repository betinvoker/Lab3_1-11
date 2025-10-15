import numpy as np
import matplotlib.pyplot as plt

def find_BMU(SOM,x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

# Функция для обновления веса ячеек SOM при наличии одного обучающего
# примера и параметры модели вместе с координатами BMU в виде кортежа
def update_weights(SOM, train_ex, learn_rate, radius_sq,
                   BMU_coord, step=3):
    g, h = BMU_coord
    # если радиус близок к нулю, то меняется только BMU
    if radius_sq < 1e-3:
        SOM[g,h,:] += learn_rate * (train_ex - SOM[g,h,:])
        return SOM
    #Замена всех ячеек в близости BMU.
    for i in range(max(0, g-step), min(SOM.shape[0], g+step)):
        for j in range(max(0, h-step), min(SOM.shape[1], h+step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM[i,j,:] += learn_rate * dist_func * (train_ex - SOM[i,j,:])
    return SOM

def train_SOM(SOM, train_data, learn_rate = .1, radius_sq = 1,
              lr_decay = .1, radius_decay = .1, epochs = 10):
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in np.arange(0, epochs):
        rand.shuffle(train_data)
        for train_ex in train_data:
            g, h = find_BMU(SOM, train_ex)
            SOM = update_weights(SOM, train_ex, 
                                 learn_rate, radius_sq, (g,h))
    # Обновление значений коэффициентов скорости обучения и радиуса
    learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
    radius_sq = radius_0 * np.exp(-epoch * radius_decay)
    return SOM

# Размеры сетки COM
m = 10
n = 10
# Количество обучающих примеров
n_x = 3000
rand = np.random.RandomState(0)
# Инициализация данных для обучения
train_data = rand.randint(0, 255, (n_x, 3))
# Инициализация SOM случайным образом
SOM = rand.randint(0, 255, (m, n, 3)).astype(float)
# Отображение обучающей матрицы и сетки SOM.
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(12, 3.5),
    subplot_kw=dict(xticks=[], yticks=[])
)
ax[0].imshow(train_data.reshape(50, 60, 3))
ax[0].title.set_text('Тренировочные данные')
ax[1].imshow(SOM.astype(int))
ax[1].tirle.set_text('Случайно инициализированная сетка SOM')