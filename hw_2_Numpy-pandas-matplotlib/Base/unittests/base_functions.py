from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m. Гарантируется что m >= 500
    Вернуть: двумерный массив, состоящий из каждого 4го элемента по оси размерности n 
    и c 120 по 500 c шагом 5 по оси размерности m
    """
    res = []
    for line in X[::4]:
        res += [[i for i in line[120:500:5]]]
    return res

# import numpy as np

# X = np.array([list(np.arange(1, 501)) for i in range(5)])
# print(get_part_of_array(X))


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    res = []
    n = len(X)
    m = len(X[0])
    d = min(n, m)
    for i in range(d):
        if X[i][i] >= 0:
            res.append(X[i][i])
    if not res:
        return -1
    else:
        return sum(res)
    
# X = np.arange(-50, 50).reshape(10, 10)
# print(X)
# print(sum_non_neg_diag(X))



def replace_values(X: List[List[float]]) -> List[List[float]]:
    """
    X - двумерный массив вещественных чисел размера n x m.
    По каждому столбцу нужно почитать среднее значение M.
    В каждом столбце отдельно заменить: значения, которые < 0.25M или > 1.5M на -1
    Вернуть: двумерный массив, копию от X, с измененными значениями по правилу выше
    """
    n = len(X)
    m = len(X[0])
    new = deepcopy(X)
    meds = []
    for j in range(m):
        val = []
        for i in range(n):
            val += [new[i][j]]
        meds += [sum(val) / n]
    for j in range(m):
        m = meds[j]
        for i in range(n):
            if new[i][j] > 1.5*m or new[i][j] < 0.25*m:
                new[i][j] = -1
    return new
                
# X = np.ones((10, 10))
# X[0, :] = 100
# X[1, :] = 10
# print(X)
# print(replace_values(X))
