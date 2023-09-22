import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


class ResolveBySklearn:
    def __init__(self,
                 x1: np.array,
                 x2: np.array) -> None:
        self.X1 = x1
        self.X2 = x2
        # Создаем метки классов
        self.y = np.array([1, 1, 1, 1, 2, 2, 2, 2])

        # Инициализируем модель LDA
        self.lda = LinearDiscriminantAnalysis(n_components=1)

        # Обучаем LDA на данных
        self.lda.fit(np.vstack((self.X1, self.X2)), self.y)

        # Проектируем данные на одномерное пространство
        self.X1_lda = self.lda.transform(x1)
        self.X2_lda = self.lda.transform(x2)

    def print(self):
        # Выводим результаты
        print("Проекции данных класса X1:")
        print(self.X1_lda)

        print("\nПроекции данных класса X2:")
        print(self.X2_lda)

    def graph(self):
        # Получаем веса (коэффициенты) линейной функции
        w = self.lda.coef_

        # Проектируем данные на одномерное пространство
        X_lda = self.lda.transform(np.vstack((self.X1, self.X2)))

        # Строим график
        plt.figure(figsize=(8, 5))
        plt.scatter(X_lda, np.zeros_like(X_lda), c=self.y, cmap='viridis', marker='o', s=100)
        plt.title('Линейная проекция данных на одномерное пространство LDA\n Sklearn')
        plt.xlabel('Линейное дискриминантное признаковое значение')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()
