import numpy as np
import typing
from collections import defaultdict


def kfold_split(
    num_objects: int, num_folds: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """

    indexes = list(np.arange(num_objects))
    step = num_objects // num_folds
    res = []
    for i in range(0, step * num_folds - step, step):
        t = (indexes[0:i] + indexes[i + step:], indexes[i:i + step])
        res.append(t)
    res.append((indexes[:step * num_folds - step], indexes[step * num_folds - step:]))
    return res


def knn_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    parameters: dict[str, list],
    score_function: callable,
    folds: list[tuple[np.ndarray, np.ndarray]],
    knn_class: object,
) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    results = dict()
    for n_neighbors in parameters["n_neighbors"]:
        for metrics in parameters["metrics"]:
            for weights in parameters["weights"]:
                for normalizer, normalizer_name in parameters["normalizers"]:
                    res = []
                    for train_ind, val_ind in folds:
                        X_train = X[train_ind]
                        y_train = y[train_ind]
                        X_valid = X[val_ind]
                        y_valid = y[val_ind]
                        if normalizer is not None:
                            normalizer.fit(X_train)
                            X_scaled = normalizer.transform(X_train)
                            X_valid_scaled = normalizer.transform(X_valid)
                        else:
                            X_scaled = X_train
                            X_valid_scaled = X_valid
                        model = knn_class(
                            n_neighbors=n_neighbors, metric=metrics, weights=weights
                        )
                        model.fit(X_scaled, y_train)
                        pred = model.predict(X_valid_scaled)
                        res.append(score_function(y_valid, pred))
                    results[(normalizer_name, n_neighbors, metrics, weights)] = np.mean(
                        res
                    )
    return results
