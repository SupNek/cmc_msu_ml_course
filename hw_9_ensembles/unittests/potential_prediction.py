import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

import numpy as np


def recenter(arr):
    non_zero_indexes = np.argwhere(arr != 0)  # получили индексы, где значение не 0
    if non_zero_indexes.size == 0:
        return arr

    center_non_zero = (
        non_zero_indexes.min(axis=0) + non_zero_indexes.max(axis=0)
    ) // 2  # нашли середину ненулевых элементов
    center_arr = np.array(arr.shape) // 2  # индексы центра массива
    shift = (
        center_arr - center_non_zero
    )  # получили смещение относительно центра массива
    new_coords = (
        non_zero_indexes + shift
    )  # получили новые координаты по которым будут размещаться данные

    result = np.zeros_like(arr)
    # заполняем результирующий массив
    for old, new in zip(non_zero_indexes, new_coords):
        if 0 <= new[0] < arr.shape[0] and 0 <= new[1] < arr.shape[1]:
            result[new[0]][new[1]] = arr[old[0]][old[1]]

    return result


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """

    def fit(self, x, y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, x, y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(x)

    def transform(self, x):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        res = []
        for mat in x:
            mat -= 20  # будем считать 20 за значение 0
            recentred_mat = recenter(mat)
            res.append(recentred_mat.reshape(-1))

        return np.array(res)


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function

    # примерно так я и представлял себе соревнования на кагле
    pt = PotentialTransformer()
    X_train = pt.fit_transform(X_train, Y_train)
    X_test = pt.transform(X_test)
    estimators = [
        (
            "extra1",
            ExtraTreesRegressor(
                n_estimators=2000,
                max_depth=10,
                criterion="friedman_mse",
                max_features="log2",
            ),
        ),
        # (
        #     "extra2",
        #     ExtraTreesRegressor(
        #         n_estimators=1000,
        #         max_depth=12,
        #         criterion="absolute_error",
        #         max_features="log2",
        #     ),
        # ),
        (
            "rf1",
            RandomForestRegressor(
                n_estimators=1000,
                max_depth=10,
                criterion="friedman_mse",
                max_features="log2",
            ),
        ),
        # (
        #     "rf2",
        #     RandomForestRegressor(
        #         n_estimators=1000,
        #         max_depth=12,
        #         criterion="absolute_error",
        #         max_features="log2",
        #         n_jobs=-1,
        #     ),
        # ),
        ("linsvr", LinearSVR(dual="auto")),
        ("svr", SVR()),
    ]
    stack = StackingRegressor(
        estimators=estimators, final_estimator=LinearSVR(dual="auto")
    )
    regressor = Pipeline(
        [("decision_tree", stack)]
    )
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
