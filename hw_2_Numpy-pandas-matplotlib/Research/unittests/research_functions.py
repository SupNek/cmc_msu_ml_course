from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    n = len(x)
    m = len(y)
    if n != m:
        return False
    return sorted(x) == sorted(y)


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    res = float('-inf')
    for i in range(len(x)-1):
        if (x[i+1] % 3 == 0) or (x[i] % 3 == 0):
            res = max(res, x[i+1] * x[i])
    if res == float('-inf'):
        return -1
    return res


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    n = len(image)
    m = len(image[0])
    c = len(image[0][0])
    res = [[0]*m for i in range(n)]
    for q in range(c):
        for j in range(m):
            for i in range(n):
                res[i][j] += image[i][j][q] * weights[q]
    return res
                
                


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    new_x = [] 
    new_y = [] 
    for line in x:
        new_x += [line[0] for i in range(line[1])]
    for line in y:
        new_y += [line[0] for i in range(line[1])]
    if len(new_x) != len(new_y):
        return -1
    res = []
    for i in range(len(new_x)):
        res.append(new_x[i] * new_y[i])
    return sum(res)


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    n = len(X)
    d = len(X[0])
    m = len(Y)
    M = [[0]*m for i in range(n)]
    for i in range(n):
        for j in range(m):
            norm_x = 0
            norm_y = 0
            for q in range(d):
                norm_x += X[i][q] ** 2
                norm_y += Y[j][q] ** 2
                M[i][j] += X[i][q] * Y[j][q]
            if (norm_x == 0) or (norm_y == 0):
                M[i][j] = 1
            else:
                M[i][j] /= (norm_x * norm_y) ** 0.5
    return M