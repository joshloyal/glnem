import numpy as np

from math import ceil
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state

from .network_utils import adjacency_to_vec


MAX_INT = np.iinfo(np.int32).max


def kfold(Y, n_splits=4, random_state=None):
    """Split dyads into k-folds.

    Parameters
    ----------
    Y : array-like, shape  (n_nodes, n_nodes)
    """
    n_nodes, _ = Y.shape
    y = adjacency_to_vec(Y)

    tril_indices = np.tril_indices(n_nodes, k=-1)
    kfolds = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train, test in kfolds.split(y):
        Y_new = np.zeros_like(Y)
        y_vec = np.copy(y)
        y_vec[test] = -1.0
        Y_new[tril_indices] = y_vec
        Y_new += Y_new.T

        yield Y_new, test
