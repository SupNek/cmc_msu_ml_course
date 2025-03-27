import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    if x.shape != y.shape:
        return False
    return np.array_equal(np.sort(x), np.sort(y))

def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if x.shape[0] < 2:
        return -1
    mod_3 = (x % 3)[1:]
    shift_mod_3 = (x % 3)[:-1]
    mul = x[1:] * x[:-1]
    new = mod_3 * shift_mod_3
    m = np.where(new == 0, mul, float('-inf')).max()
    if m == float('-inf'):
        return -1
    return int(m)

# x = np.array([15])
# x = np.array([56, 2, 55,  6, 89, 70, 38, 22, 16, 36, 29, 73, 46, 65, 87, 84, 64, 83, 38, 44, 30, 48, 66, 87,
#  65, 59, 14, 83, 16, 90, 91, 75, 69, 11, 76, 30, 28, 71, 87, 58, 80, 94, 61, 93, 32, 92, 97,  4,
#  61, 88, 60, 84])
# print(max_prod_mod_3(x)) # 8316

def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return (image * weights.reshape(1, 1, -1)).sum(axis=2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    
    if x[:, 1].sum() != y[:, 1].sum():
        return -1
    def decode_rle(x):
        a = x[:, 0]
        b = x[:, 1]
        lenght = b.max()
        c = np.tile(np.arange(lenght), (b.shape[0], 1)) - b.reshape(-1, 1) # подготовка для сбора маски
        mask = np.where(c < 0, 1, float('inf'))
        norm_x = mask * a.reshape(-1, 1)
        norm_x = norm_x.ravel()[np.where(norm_x.ravel() != float('inf'))]
        return norm_x
    new_x = decode_rle(x)
    new_y = decode_rle(y)
    return int(new_x.dot(new_y))
    
# x = np.array([[1, 2], [2, 3], [3, 1]])
# y = np.array([[1, 1], [0, 5]])
# print(rle_scalar(x, y))


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    sx = np.sqrt(np.sum(X**2, axis=1, keepdims=True))
    sy = np.sqrt(np.sum(Y**2, axis=1, keepdims=True))
    norm = sx*sy.T
    res = X.dot(Y.T) / np.where(norm == 0, 1, norm)
    return np.where(norm != 0, res, 1)

# X = np.array([[1, 0, 1], [0, 1, 1], [0, 1, 1]])
# Y = np.array([[0, 0, 0], [1, 0, 0]])
# print(cosine_distance(X, Y))