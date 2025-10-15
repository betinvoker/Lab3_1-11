
import sys
import datetime
import argparse

class Tee:
    """Класс для дублирования вывода в консоль и файл"""
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    
    def write(self, data):
        self.stdout.write(data)  # В консоль
        self.file.write(data)    # В файл
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

def main():
    # Использование
    tee = Tee(f'log-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.txt')
    sys.stdout = tee

    try:
        print("Этот текст будет и в консоли, и в файле")
        user_input = input("Введите что-нибудь: ")
        print(gree + 2)
        print(f"Вы ввели: {user_input}")
    except Exception as e:
        print(f"Ошибка: {e}")

    tee.close()
    sys.stdout = tee.stdout  # Возвращаем стандартный вывод

if __name__ == "__main__":
    main()