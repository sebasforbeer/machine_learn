import numpy as np
import matplotlib.pyplot as plt


class ResolveByStandart:
    def __init__(self,
                 x1: np.array,
                 x2: np.array) -> None:
        self.X1 = x1
        self.X2 = x2

        # Вычисляем средние значения признаков для каждого класса
        mean_X1 = np.mean(self.X1, axis=0)
        mean_X2 = np.mean(self.X2, axis=0)

        # Вычисляем матрицу разброса внутри каждого класса (внутриклассовая матрица)
        S_W = (np.dot((self.X1 - mean_X1).T, (self.X1 - mean_X1))
               + np.dot((self.X2 - mean_X2).T, (self.X2 - mean_X2)))

        # Вычисляем разницу средних значений между классами
        mean_diff = (mean_X1 - mean_X2).reshape(-1, 1)

        # Вычисляем вектор весов, который максимизирует критерий Фишера
        self.w = np.dot(np.linalg.inv(S_W), mean_diff)

        # Проектируем данные на одномерное пространство
        self.X1_lda = np.dot(self.X1, self.w)
        self.X2_lda = np.dot(self.X2, self.w)

    def print(self):
        # Выводим результаты
        print("Проекции данных класса X1:")
        print(self.X1_lda)

        print("\nПроекции данных класса X2:")
        print(self.X2_lda)

    def graph(self):
        # Строим график
        plt.figure(figsize=(8, 5))
        plt.scatter(self.X1_lda, np.zeros_like(self.X1_lda), label='X1', marker='o', s=100)
        plt.scatter(self.X2_lda, np.zeros_like(self.X2_lda), label='X2', marker='s', s=100)
        plt.legend(loc='best')
        plt.title('Проекции данных на одномерное пространство LDA\n standart')
        plt.xlabel('Линейное дискриминантное признаковое значение')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()
