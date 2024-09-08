import numpy as np
import pandas as pd
import networkx as nx
import joblib

from os.path import dirname, join
from glnem.network_utils import adjacency_to_vec


__all__ = ['load_banana_trade']
    

def load_banana_trade(max_nodes=None, include_diag=False):
    module_path = dirname(__file__)

    base_path = join(module_path, 'raw_data', 'BACI_HS17_V202301')
    if include_diag:
        A = joblib.load(join(base_path, f'banana_diag.npy'))
        countries = pd.read_csv(
                join(base_path, f'countries_diag.csv')).values[:, -1]
    else:
        A = joblib.load(join(base_path, f'banana.npy'))
        countries = pd.read_csv(
                 join(base_path, f'countries_banana.csv')).values
    Y = A[0]

    if max_nodes is not None:
        node_ids = np.argsort(Y.sum(axis=1))[::-1][:max_nodes]
        Y = Y[node_ids][:, node_ids]
        countries = countries[node_ids]

    n_nodes = Y.shape[0]
    if include_diag:
        n_dyads = int(0.5 * n_nodes * (n_nodes + 1))
    else:
        n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    n_features = A.shape[0] - 1
    X = np.zeros((n_dyads, n_features))
    for p in range(n_features):
        Ap = A[p+1]
        if max_nodes is not None:
            Ap = Ap[node_ids][:, node_ids]
        X[:, p] = adjacency_to_vec(Ap, include_diag=include_diag)

    return Y, X, countries
