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

def predict(SOM, data):
    predictions = []
    for input_vector in data:
        bmu_index = find_BMU(SOM, input_vector)
        predictions.append(bmu_index)
    return np.array(predictions)

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
ax[1].title.set_text('Случайно инициализированная сетка SOM')

fig, ax = plt.subplots(
    nrows=1, ncols=4, figsize=(15, 3.5),
    subplot_kw=dict(xticks=[], yticks=[])
)

total_epochs = 0

for epochs, i in zip([1,4,5,10], range(0,4)):
    total_epochs += epochs
    SOM = train_SOM(SOM, train_data, epochs=epochs, learn_rate=0.3, radius_sq=10)
    ax[i].imshow(SOM.astype(int))
    ax[i].title.set_text('Epochs = ' + str(total_epochs))

fig, ax = plt.subplots(
    nrows=3, ncols=3, figsize=(15, 15),
    subplot_kw=dict(xticks=[], yticks=[])
)

# Инициализируем SOM случайным образом
# в одном и том же состоянии (рандом происходит однократно)
for learn_rate, i in zip([0.001,0.5,0.99], [0,1,2]):
    for radius_sq, j in zip([0.01,1,10], [0,1,2]):
        rand = np.random.RandomState(0)
        SOM = rand.randint(0, 255, (m,n,3)).astype(float)
        SOM = train_SOM(SOM, train_data, epochs=5,
                        learn_rate=learn_rate,
                        radius_sq=radius_sq)
        ax[i][j].imshow(SOM.astype(int))
        ax[i][j].title.set_text('$\eta$ = ' + str(learn_rate) +
                                ', $\sigma^2$ = ' + str(radius_sq))

SOM1 = rand.randint(0, 255, (m,n,3)).astype(float)
SOM1 = train_SOM(SOM1, train_data, epochs=20, learn_rate = 0.3, 
                    radius_sq = 10)
fig, ax = plt.subplots(
    nrows=1, ncols=1, figsize=(5, 5),
    subplot_kw=dict(xticks=[0,1,2,3,4,5,6,7,8,9,10], 
                    yticks=[10,9,8,7,6,5,4,3,2,1,0])
)
ax.imshow(SOM1.astype(int))
ax.title.set_text('Epochs = 20')

arr = np.array([
    [69, 100, 65],
    [93, 113, 92],
    [189, 50, 157],
    [164, 124, 54],
    [39, 87, 202],
    [221, 198, 65],
    [30, 216, 80],
    [87, 116, 7],
    [55, 194, 142],
    [45, 103, 59]
])

predictions = predict(SOM1, arr)

print("Тестовые (новые) данные:")
print(arr)
print("\nПредсказанный кластер (координаты BMU) для новых данных:")
print(predictions)

plt.style.use('_mpl-gallery')

width=5
height=5
plt.figure(figsize=(width, height))
for input_vector in train_data:
     bmu_coords = find_BMU(SOM1, input_vector)
     plt.plot(bmu_coords[0], bmu_coords[1], 'ro', markersize=2,)

plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(figsize=(8, 8))
for vector in train_data:
    bmu_coords = find_BMU(SOM1, vector)
    ax.plot(bmu_coords[1], bmu_coords[0], 'o', color='pink', markersize=4, zorder=1)

for vector in arr:
    bmu_coords = predict(SOM1, [vector])[0]
    ax.plot(bmu_coords[1], bmu_coords[0], 'o', color='green', markersize=8, zorder=2, label='Тестовые точки')

texts_at_coords = {}
for i, pred in enumerate(predictions):
    coord_tuple = tuple(pred) # (y, x)
    label = f' new point {i}\n {list(coord_tuple)}'
    if coord_tuple in texts_at_coords:
        texts_at_coords[coord_tuple] += f"\n{label}"
    else:
        texts_at_coords[coord_tuple] = label

for coords, text in texts_at_coords.items():
    ax.text(coords[1] + 0.1, coords[0] + 0.1, text, fontsize=9, ha='left', va='bottom')

ax.set_title('Самоорганизующаяся карта Кохонена')
ax.set_xlim(-0.5, SOM1.shape[1] - 0.5) 
ax.set_ylim(-0.5, SOM1.shape[0] - 0.5) 
ax.invert_yaxis() 

plt.show()