import numpy as np
import pandas as pd

from math import ceil
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from .glnem import GLNEM
from .network_utils import adjacency_to_vec


MAX_INT = np.iinfo(np.int32).max


def kfold(Y, n_splits=4, random_state=None):
    """Split dyads into k-folds.

    Parameters
    ----------
    Y : array-like, shape  (n_nodes, n_nodes)
    X : array-like, shape  (n_dyads, n_features)
    """
    n_nodes, _ = Y.shape
    y = adjacency_to_vec(Y)

    tril_indices = np.tril_indices(n_nodes, k=-1)
    kfolds = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train, test in kfolds.split(y):
        Y_train = np.zeros_like(Y)
        y_vec = np.copy(y)
        y_vec[test] = -1.0
        Y_train[tril_indices] = y_vec
        Y_train += Y_train.T
        
        yield Y_train, test


def kfold_selection_single(Y, X, family, link, n_features, 
        n_warmup=500, n_samples=500, n_folds=4, random_state=42):
    n_nodes = Y.shape[0]
    loglik = 0.
    folds = kfold(Y, n_splits=n_folds, random_state=random_state)
    for Y_train, test_indices in folds:
        # fit model
        model = GLNEM(
            family=family, link=link,
            infer_dimension=False,
            infer_sigma=False,
            n_features=n_features,
            random_state=123)

        model.sample(Y_train, X=X, n_warmup=n_warmup, n_samples=n_samples)

        loglik += model.loglikelihood(Y, test_indices=test_indices)

    return n_features, loglik / n_folds


def kfold_selection(Y, X=None, family='bernoulli', link='logit',
        min_features=1, max_features=10, 
        n_warmup=500, n_samples=500, n_folds=4, n_jobs=-1, random_state=42):

    res = Parallel(n_jobs=n_jobs)(delayed(kfold_selection_single)(
        Y=Y, X=X, family=family, link=link, n_features=d,
        n_warmup=n_warmup, n_samples=n_samples, 
        n_folds=n_folds, random_state=random_state) for 
            d in range(min_features, max_features + 1))

    return pd.DataFrame(np.asarray(res), columns=['n_features', 'loglik'])


def ic_selection_single(Y, X, family, link, n_features, 
        n_warmup=500, n_samples=500):

    # fit model
    model = GLNEM(
            family=family, link=link,
            infer_dimension=False, 
            infer_sigma=False,
            n_features=n_features,
            random_state=123)

    model.sample(Y, X=X, n_warmup=n_warmup, n_samples=n_samples)

    return n_features, model.waic(), model.dic(), model.aic(), model.bic()


def ic_selection(Y, X=None, family='bernoulli', link='logit',
        min_features=1, max_features=10, 
        n_warmup=500, n_samples=500, n_jobs=-1):

    res = Parallel(n_jobs=n_jobs)(delayed(ic_selection_single)(
        Y=Y, X=X, family=family, link=link, n_features=d,
        n_warmup=n_warmup, n_samples=n_samples) for 
            d in range(min_features, max_features + 1))

    return pd.DataFrame(np.asarray(res), 
            columns=['n_features', 'waic', 'dic', 'aic', 'bic'])
