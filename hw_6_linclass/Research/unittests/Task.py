import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.newcols = {}

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        for column in X.columns:
            self.newcols[column] = sorted(X[column].unique())

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        n_features = sum(len(categories) for categories in self.newcols.values())
        result = np.zeros((X.shape[0], n_features), dtype=self.dtype)

        start_idx = 0
        for column, categories in self.newcols.items():
            for i, category in enumerate(categories):
                indices = X[column] == category
                result[indices, start_idx + i] = 1
            start_idx += len(categories)

        return result

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.categories = {}

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        for col in X.columns:
            objs = {}
            for el in X[col].unique():
                indexes = X[col] == el
                successes = Y[indexes].mean()
                counters = indexes.mean()
                objs[el] = (successes, counters)
            self.categories[col] = objs

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        result = np.zeros((X.shape[0], 3 * X.shape[1]), dtype=self.dtype)

        start_idx = 0
        for column, categories in self.categories.items():
            for el, values in categories.items():
                indices = X[column] == el
                result[indices, start_idx] = values[0]
                result[indices, start_idx + 1] = values[1]
                result[indices, start_idx + 2] = (values[0] + a) / (values[1] + b)
            start_idx += 3
        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack(
            (idx[: i * n_], idx[(i + 1) * n_:])
        )
    yield idx[(n_splits - 1) * n_:], idx[: (n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.fold_counters = []

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        for fold_ind, other_ind in group_k_fold(X.shape[0], self.n_folds, seed):
            new_fold = {}
            X_fold, Y_fold = X.iloc[other_ind], Y.iloc[other_ind]
            for col in X.columns:
                new_fold[col] = {}
                for el in X[col].unique():
                    indexes = X_fold[col] == el
                    new_fold[col][el] = (Y_fold[indexes].mean(), indexes.mean())
            self.fold_counters.append((fold_ind, new_fold))

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        res = np.zeros((X.shape[0], 3 * X.shape[1]), dtype=self.dtype)

        for fold_idx, fold_counter in self.fold_counters:
            i = 0
            for col in X.columns:
                for j in fold_idx:
                    value = X.iloc[j, i // 3]
                    counters = fold_counter[col][value]
                    res[j, i] = counters[0]
                    res[j, i + 1] = counters[1]
                    res[j, i + 2] = (counters[0] + a) / (counters[1] + b)
                i += 3
        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    un_val = np.unique(x)
    w = np.zeros(un_val.shape)
    for i, elem in enumerate(un_val):
        w[i] = sum(y[x == elem]) / sum(x == elem)
    return w
