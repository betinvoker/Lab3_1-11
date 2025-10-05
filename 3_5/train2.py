import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
root_mean_squared_error, mean_absolute_percentage_error, r2_score

# Загрузка данных
engagement_data = pd.read_csv('engagement_data.csv')

# Подготовка данных для Prophet
engagement_data['datetime'] = pd.to_datetime(engagement_data['datetime'])
engagement_data = engagement_data.rename(columns={'datetime': 'ds',
                                                   'engagement': 'y'})

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

# Объединение фактических и прогнозируемых значений для расчета метрик
actual = engagement_data['y']
predicted = forecast1['yhat'].head(len(actual))
# Расчет метрик
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = root_mean_squared_error(actual, predicted)
mape = mean_absolute_percentage_error(actual, predicted)
r2 = r2_score(actual, predicted)
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")
print(f"r2: {r2}")

# Создание будущих дат для прогноза на 7 дней с почасовым шагом
# Параметр include_history=False указывает, что не включаются исторические данные в прогноз
future2 = model1.make_future_dataframe(periods=7*24, freq='h',
                    include_history=False) # 7 дней * 24 часа = 168 часов
# Прогнозирование
forecast2 = model1.predict(future2)

# Визуализация прогноза
fig1 = model1.plot(forecast2)
plt.title('Прогноз активностей пользователей')
plt.show()

# Визуализация компонентов прогноза
fig2 = model1.plot_components(forecast2)
plt.title('Составляющие компоненты прогноза')
plt.show()

print(forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(24)) # Вывод первых 24 строк (1 день)

top_hours = forecast2[['ds', 'yhat']].sort_values(by='yhat', ascending=False).head(10)
print('Лучшие часы для публикации:')
print(top_hours)

