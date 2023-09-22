import numpy as np

from PZ2.ResolveByStandart import ResolveByStandart
from PZ2.ResolveBySklearn import ResolveBySklearn


class PZ2:
    def __init__(self):
        # Задаем данные для классов X1 и X2
        self.X1 = np.array([[-6, 4], [-8, 2], [-9, 4], [-2, 5]])
        self.X2 = np.array([[2, -5], [7, -4], [5, -3], [6, -2]])

    def resolve_by_sklearn(self):
        rbs = ResolveBySklearn(self.X1, self.X2)
        rbs.print()
        rbs.graph()

    def resolve_by_standart(self):
        rbs = ResolveByStandart(self.X1, self.X2)
        rbs.print()
        rbs.graph()
