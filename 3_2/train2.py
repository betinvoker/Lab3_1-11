import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('grain_sales.csv')

df = pd.get_dummies(df, columns=['Канал_продаж', 'Сезон', 'Регион', 'Тип_покупателя'])

X = df[['Цена_за_тонну']]
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

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Фактические данные')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Линия регрессии')
# Добавим заголовок и подписи
plt.title('Простая линейная регрессия: Объём продаж от Цены за тонну', fontsize=14)
plt.xlabel('Цена за тонну (руб.)', fontsize=12)
plt.ylabel('Объём продаж (тонн)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

print(f"Коэффициент (наклон): {model.coef_[0]:.6f}")
print(f"Свободный член (смещение): {model.intercept_:.3f}")