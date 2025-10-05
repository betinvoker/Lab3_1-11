import pandas as pd
import numpy as np
import random
import shap
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import scipy.stats as status
import seaborn as sns

def generate_dates(start_date='2023-02-01', days=1000):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

channels = ['бот', 'почта']
seasons_by_month = {
    '01': 'зима', '02': 'зима', '03': 'весна',
    '04': 'весна', '05': 'весна', '06': 'лето',
    '07': 'лето', '08': 'лето', '09': 'осень',
    '10': 'осень', '11': 'осень', '12': 'зима'
}
regions = ['Центральный', 'Южный', 'Уральский', 'Сибирский']
customer_types = ['опт', 'розница']

dates = generate_dates(days=1000)
price_per_ton = np.random.randint(12000, 15000, size=1000) # цена за тонну, случайное целое число от 12000 до 15000 рублей
channel = [random.choice(channels) for _ in range(1000)]
season = [seasons_by_month[date.split('-')[1]] for date in dates]
promotions = [random.choice([0, 1]) for _ in range(1000)] # флаг проведения акции: 0 (нет) или 1 (да).

promotions = np.array(promotions)
channel = np.array(channel)
season = np.array(season)

sales_amount = (
    30
    - 0.001 * (price_per_ton - 13500) # влияние цены
    + 5 * promotions # влияние акций
    + np.where(season == 'лето', 4, 0) # влияние сезона
    + np.random.normal(0, 2, size=1000) # шум
)
sales_amount = np.clip(sales_amount, 5, 50) # ограничиваем диапазон
sales_amount = np.round(sales_amount, 1) # округляем до 0.1 тонны
price_per_ton = np.random.randint(12000, 15000, size=1000) # цена за тонну, случайное целое число от 12000 до 15000 рублей
channel = [random.choice(channels) for _ in range(1000)]
season = [seasons_by_month[date.split('-')[1]] for date in dates]
promotions = [random.choice([0, 1]) for _ in range(1000)] # флаг проведения акции: 0 (нет) или 1 (да).
region = [random.choice(regions) for _ in range(1000)]
customer_type = [random.choice(customer_types) for _ in range(1000)]
season = np.array([seasons_by_month[date.split('-')[1]] for date in dates])
promotions = np.array([random.choice([0, 1]) for _ in range(1000)])

df = pd.DataFrame({
    'Дата': dates,
    'Объем_продаж': sales_amount,
    'Цена_за_тонну': price_per_ton,
    'Канал_продаж': channel,
    'Сезон': season,
    'Акции': promotions,
    'Регион': region,
    'Тип_покупателя': customer_type
})

print(df)

df.to_csv('grain_sales1.csv', index=False, encoding='utf-8-sig')
print("Файл grain_sales1.csv успешно создан!")

df = pd.read_csv('grain_sales1.csv')
df = pd.get_dummies(df, columns=['Канал_продаж', 'Сезон', 'Регион', 'Тип_покупателя'])

X = df.drop(['Объем_продаж', 'Дата'], axis=1)
y = df['Объем_продаж']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5 # Корень из MSE — это RMSE
r2 = r2_score(y_test, y_pred)
print(f"\nRMSE: {round(rmse, 2)} тонн")
print(f"MSE: {round(mse, 2)} в квадрате тонн")
print(f"R²: {round(r2, 4)}")

new_data = pd.DataFrame([{
    'Сезон': 'весна',
    'Регион': 'Центральный',
    'Тип_покупателя': 'опт',
    'Канал_продаж': 'бот',
    'Цена_за_тонну': 13500,
    'Акции': 1
}])

# Преобразуем категориальные переменные в dummy-переменные
new_data_encoded = pd.get_dummies(new_data, columns=['Канал_продаж', 'Сезон', 'Регион', 'Тип_покупателя'])
feature_columns = X.columns
new_data_encoded = new_data_encoded.reindex(columns=feature_columns, fill_value=0)

prediction = model.predict(new_data_encoded)
print(f"\nПрогнозируемый объем продаж: {prediction[0]:.1f} тонн")

# График 1. Фактические vs Предсказанные значения (Actual vs Predicted)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
# Добавим заголовок и подписи
plt.title('Фактические vs Предсказанные', fontsize=14)
plt.xlabel('Фактические значения', fontsize=12)
plt.ylabel('Предсказанные значения', fontsize=12)
plt.grid(True)
plt.show()

# График 2. График остатков
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
# Добавим заголовок и подписи
plt.title('График остатков', fontsize=14)
plt.xlabel('Предсказанные значения', fontsize=12)
plt.ylabel('Остатки', fontsize=12)
plt.grid(True)
plt.show()

# График 3. Гистограмма остатков + Q-Q plot
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
sns.histplot(regions, kde=True)
plt.title('Распределение остатков', fontsize=14)
plt.xlabel('Остатки', fontsize=12)

# Q-Q plot
plt.subplot(1,2,2)
status.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q график остатков')

plt.tight_layout()
plt.show()

# График 4. Влияние каждого признака: коэффициенты модели (Feature Importance)
coef_df = pd.DataFrame({
    'Признак': feature_columns,
    'Коэффициент': model.coef_
}).sort_values('Коэффициент', key=abs, ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(
    data=coef_df,
    x='Коэффициент',
    y='Признак',
    hue='Признак',
    palette='coolwarm',
    legend=False
)
plt.title('Влияние признаков на обём продаж (коэффициенты модели)')
plt.xlabel('Значение коэффициента')
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# График 5: Partial Dependence Plot (PDP) - частная зависимость
features_to_plot = [0,1,2]
PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features_to_plot,
    feature_names=feature_columns,
    grid_resolution=50
)
plt.tight_layout()
plt.show()

# График 6: SHAP-графики (SHapley Additive exPlanations) - продвинутая интерпретация
# Проверим и проведём типы данных
print("Типы данных в X_train до преобразования:")
print(X_train.dtypes)

# Приводим к float64 - SHAP требует числовые массивы
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

print("\nТипы данных в X_train после преобразования:")
print(X_train.dtypes)

# Создаем explainer и вычисляем SHAP-значения
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# График водопада для первого наблюдения
shap.plots.waterfall(shap_values[0])

# Summary plot - важность и влияние признаков
shap.summary_plot(shap_values, X_test, feature_names=feature_columns)

# График 7: График влияния категориальных признаков (через группировку)
df = pd.read_csv('grain_sales.csv')

df_raw = df.copy() # Делайте это сразу после создания df, до get_dummies!
print(df_raw.columns.tolist())
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(data=df_raw, x='Сезон', y='Объем_продаж')
plt.title('Объем продаж по сезонам')
plt.grid(True)

plt.subplot(1,2,2)
sns.boxplot(data=df_raw, x='Канал_продаж', y='Объем_продаж')
plt.title('Объем продаж по каналам')
plt.grid(True)

plt.tight_layout()
plt.show()