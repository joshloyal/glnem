import numpy as np
import pandas as pd

from math import ceil
from sklearn.base import clone
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.utils import check_random_state
from joblib import Parallel, delayed

from .glnem import GLNEM
from .network_utils import adjacency_to_vec


MAX_INT = np.iinfo(np.int32).max


def train_test_split(Y, test_size=0.2, random_state=None):
    n_nodes, _ = Y.shape
    y = adjacency_to_vec(Y)

    tril_indices = np.tril_indices(n_nodes, k=-1)
    rs = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state)
    train, test = next(rs.split(y))
 
    return train, test


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
        #Y_train = np.zeros_like(Y)
        #y_vec = np.copy(y)
        #y_vec[test] = -1.0
        #Y_train[tril_indices] = y_vec
        #Y_train += Y_train.T
        
        #yield Y_train, test
        yield test


def kfold_selection_single(Y, X, family, link, n_features, 
        n_warmup=500, n_samples=500, n_folds=4, random_state=42):
    n_nodes = Y.shape[0]
    logliks = np.zeros(n_folds)
    folds = kfold(Y, n_splits=n_folds, random_state=random_state)
    #for k, (Y_train, test_indices) in enumerate(folds):
    for k, test_indices in enumerate(folds):
        # fit model
        model = GLNEM(
            family=family, link=link,
            infer_dimension=False,
            n_features=n_features,
            random_state=42)

        #model.sample(Y_train, X=X, n_warmup=n_warmup, n_samples=n_samples)
        
        model.sample(Y, X=X, n_warmup=n_warmup, n_samples=n_samples,
            missing_edges=test_indices)

        logliks[k] = model.loglikelihood(Y, test_indices=test_indices)

    return n_features, np.mean(logliks), np.std(logliks)


def kfold_selection(Y, X=None, family='bernoulli', link='logit',
        min_features=1, max_features=10, 
        n_warmup=500, n_samples=500, n_folds=4, n_jobs=-1, random_state=42):

    res = Parallel(n_jobs=n_jobs)(delayed(kfold_selection_single)(
        Y=Y, X=X, family=family, link=link, n_features=d,
        n_warmup=n_warmup, n_samples=n_samples, 
        n_folds=n_folds, random_state=random_state) for 
            d in range(min_features, max_features + 1))

    return pd.DataFrame(np.asarray(res), columns=['n_features', 'loglik', 'se'])


def ic_selection_single(Y, X, family, link, n_features, 
        n_warmup=500, n_samples=500):

    # fit model
    model = GLNEM(
            family=family, link=link,
            infer_dimension=False, 
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
