import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение данных из файла
filename = "output.csv"  # Укажите путь к вашему файлу
with open(filename, 'r') as file:
    data = file.readlines()

# Преобразование строк в числовую матрицу
matrix = []
for line in data:
    row = [float(x.strip()) for x in line.split(',') if x.strip()]
    matrix.append(row)

# Преобразование в numpy-массив
temperature_matrix = np.matrix(matrix, dtype=np.float32)


# Построение тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(temperature_matrix, annot=False, fmt=".1e", cmap="coolwarm", cbar_kws={'label': 'Температура'})
plt.title("Тепловая карта температур")
plt.xlabel("X координата")
plt.ylabel("Y координата")
plt.show()