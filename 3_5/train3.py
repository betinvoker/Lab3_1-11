import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
root_mean_squared_error, mean_absolute_percentage_error, r2_score

# Функция для обучения и оценки модели по одному типу контента
def train_and_evaluate(content_type):
    print(f"\nОбработка типа контента: {content_type}")
    df = data[data['content_type'] == content_type].copy()
    # Агрегируем по часу, дню недели и часу суток
    df_grouped = df.groupby(['ds', 'hour', 'day_of_week']).agg({'y': 'mean'}).reset_index()
    # Разделяем на train и test (последние 7 дней — тест)
    split_date = df_grouped['ds'].max() - pd.Timedelta(days=7)
    train_df = df_grouped[df_grouped['ds'] <= split_date]
    test_df = df_grouped[df_grouped['ds'] > split_date]
    # Инициализация модели Prophet с праздниками
    model = Prophet(
        holidays=holidays,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    # Добавляем регрессоры
    model.add_regressor('hour')
    model.add_regressor('day_of_week')

    # Обучение
    model.fit(train_df.rename(columns={'y': 'y'}))
    # Подготовка данных для прогноза на тесте
    future = test_df[['ds', 'hour', 'day_of_week']].copy()
    # Прогноз
    forecast = model.predict(future)
    # Метрики
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    #r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"r2: {r2:.2f}")
    # Прогноз на ближайшие 7 дней
    future_dates = pd.date_range(start=df_grouped['ds'].max() +
    pd.Timedelta(hours=1), periods=24*7, freq='h')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['hour'] = future_df['ds'].dt.hour
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek
    forecast_future = model.predict(future_df)

    # Возвращаем прогноз и модель для дальнейшего использования
    return forecast_future, model

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

# Размер данных (например, за 5 месяцев с почасовой частотой)
data_size = 24 * 7 * 22 * 5 # примерно 5 недель * 7 дней * 24 часа * 5 месяцев (примерно)
# Генерируем временные метки с почасовым шагом
date_range = pd.date_range(start='2025-01-01', periods=data_size, freq='h')
np.random.seed(42)

# Генерация признаков
data = {
    'datetime': date_range,
    'hour': date_range.hour,
    'day_of_week': date_range.dayofweek,
    'content_type': np.random.choice(['text', 'image', 'video'],
                        size=data_size, p=[0.5, 0.3, 0.2]),
    'likes': np.random.poisson(lam=20, size=data_size), # распределение Пуассона для лайков
    'comments': np.random.poisson(lam=5, size=data_size), # для комментариев
    'shares': np.random.poisson(lam=2, size=data_size), # для репостов 
}
df = pd.DataFrame(data)

# Можно добавить общую вовлечённость как сумму или взвешенную сумму
df['engagement'] = df['likes'] + df['comments'] * 2 + df['shares'] * 3
# Сохраняем в CSV
df.to_csv('engagement_data_extended.csv', index=False)
print(df.head())

# Загрузка данных
data = pd.read_csv('engagement_data_extended.csv', parse_dates=['datetime'])

# Подготовка данных для Prophet
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.rename(columns={'datetime': 'ds',
                                'engagement': 'y'})

# Добавляем временные признаки
# data['ds'] = data['datetime'].dt.floor('h')
data['hour'] = data['ds'].dt.hour
data['day_of_week'] = data['ds'].dt.dayofweek

# Список типов контента
content_types = data['content_type'].unique()

# Словарь для хранения прогнозов по типам контента
forecasts = {}
models = {}
for ctype in content_types:
    forecast, model = train_and_evaluate(ctype)
    forecasts[ctype] = forecast
    models[ctype] = model

# Анализируем лучшие часы для публикации по каждому типу контента
for ctype, forecast in forecasts.items():
    forecast['hour'] = forecast['ds'].dt.hour
    best_hours = forecast.groupby('hour')['yhat'].mean().sort_values(ascending=False)
    print(f"Лучшие часы для публикации для типа контента '{ctype}':")
    print(best_hours.head(5))