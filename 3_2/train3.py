import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('grain_sales.csv')

df = pd.get_dummies(df, columns=['Канал_продаж', 'Сезон', 'Регион', 'Тип_покупателя'])

X = df.drop(['Объем_продаж', 'Дата'], axis=1)
y = df['Объем_продаж']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Вывод коэффициентов модели
print("\n" + "="*60)
print("КОЭФФИЦИЕНТЫ МОДЕЛИ ЛИНЕЙНОЙ РЕГРЕССИИ")
print("="*60)
# Создаем DataFrame для удобного отображения
coef_df = pd.DataFrame({
    'Признак': X.columns,
    'Коэффициент': model.coef_
})
# Добавляем свободный член в конец
coef_df = pd.concat([
coef_df,
pd.DataFrame([{'Признак': 'Свободный член (intercept)', 'Коэффициент':
model.intercept_}])
], ignore_index=True)
# Сортируем по модулю коэффициентов — чтобы видеть самые сильные влияния
coef_df['Абсолютное_влияние'] = coef_df['Коэффициент'].abs()
coef_df_sorted = coef_df.sort_values(by='Абсолютное_влияние', ascending=False)
# Выводим
print(coef_df_sorted.to_string(index=False,
float_format='{:,.4f}'.format))

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5 # Корень из MSE — это RMSE
r2 = r2_score(y_test, y_pred)
print(f"\nRMSE: {round(rmse, 2)} тонн")
print(f"MSE: {round(mse, 2)} в квадрате тонн")
print(f"R²: {round(r2, 4)}")
