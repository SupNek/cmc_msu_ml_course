import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    _, counts = np.unique(sample, return_counts=True)
    probability = counts / sum(counts)
    gini = float(1 - sum(probability**2))
    entropy = float(-sum(probability * np.log(probability)))
    error = float(1 - max(probability))
    return {"gini": gini, "entropy": entropy, "error": error}
