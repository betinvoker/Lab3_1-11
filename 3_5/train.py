import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

date_range = pd.date_range(start='2025-01-01 00:00:00', 
                           end='2025-05-28 23:00:00', freq='h')

np.random.seed(42)
base_engagement = 50

hourly_pattern = np.array([10, 8, 6, 5, 4, 3, 5, 10, 20, 30, 40,
                            50, 60, 70, 80, 90, 85, 75, 60, 50,
                            40, 30, 20, 15])

dayofweek_pattern = np.array([0.8, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7])

engagement_data = pd.DataFrame({'datetime': date_range})
engagement_data['hour'] = engagement_data['datetime'].dt.hour
engagement_data['dayofweek'] = engagement_data['datetime'].dt.dayofweek

engagement_data['engagement'] = base_engagement + \
    hourly_pattern[engagement_data['hour'].values] * \
    dayofweek_pattern[engagement_data['dayofweek'].values] + \
    np.random.normal(0, 5, len(engagement_data)) # шум
engagement_data['engagement'] = engagement_data['engagement'].round().astype(int)

engagement_data = engagement_data[['datetime', 'engagement']]

engagement_data.to_csv('engagement_data.csv', index=False)

print(engagement_data.head())

# Загрузка данных
engagement_data = pd.read_csv('engagement_data.csv')

# Подготовка данных для Prophet
engagement_data['datetime'] = pd.to_datetime(engagement_data['datetime'])
engagement_data = engagement_data.rename(columns={'datetime': 'ds',
                                                   'engagement': 'y'})

# Создание и обучение модели Prophet
model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(engagement_data)

# Создание будущих дат для прогноза
# Прогноз на следующие 24 часа
future = model.make_future_dataframe(periods=24, freq='h')

# Прогнозирование
forecast = model.predict(future)

# Визуализация прогноза
fig1 = model.plot(forecast)
plt.title('Прогноз активностей пользователей')
plt.show()

# Визуализация компонентов прогноза
fig2 = model.plot_components(forecast)
plt.title('Составляющие компоненты прогноза')
plt.show()

# Создание DataFrame с праздничными днями
holidays = pd.DataFrame({
    'holiday': 'public_holiday',
    'ds': pd.to_datetime([
        '2025-01-01', # Новый год
        '2025-01-07', # Рождество
        '2025-02-23', # День защитника Отечества
        '2025-03-08', # Международный женский день
        '2025-05-01', # Праздник Весны и Труда
        '2025-05-09'  # День Победы
    ]),
    'lower window': -1,
    'upper window': 1
})

# Создание и обучение модели Prophet с учетом праздников
model1 = Prophet(daily_seasonality=True, weekly_seasonality=True,
                 holidays=holidays)
model1.fit(engagement_data)

# Создание будущих дат для прогноза на 7 дней с почасовым шагом
future1 = model1.make_future_dataframe(periods=7*24, freq='h') # 7 дней * 24 часа = 168 часов

# Прогнозирование
forecast1 = model1.predict(future1)

# Визуализация прогноза
fig1 = model1.plot(forecast1)
plt.title('Прогноз активностей пользователей')
plt.show()

# Визуализация компонентов прогноза
fig2 = model1.plot_components(forecast1)
plt.title('Составляющие компоненты прогноза')
plt.show()