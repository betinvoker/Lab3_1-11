import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

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
sales_amount = np.round(np.random.uniform(5, 50, size=1000), 1) # случайное значение от 5 до 50 с округлением до 1 знака после запятой.
price_per_ton = np.random.randint(12000, 15000, size=1000) # цена за тонну, случайное целое число от 12000 до 15000 рублей
channel = [random.choice(channels) for _ in range(1000)]
season = [seasons_by_month[date.split('-')[1]] for date in dates]
promotions = [random.choice([0, 1]) for _ in range(1000)] # флаг проведения акции: 0 (нет) или 1 (да).
region = [random.choice(regions) for _ in range(1000)]
customer_type = [random.choice(customer_types) for _ in range(1000)]

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

df.to_csv('grain_sales.csv', index=False, encoding='utf-8-sig')
print("Файл grain_sales.csv успешно создан!")