import numpy as np
import pandas as pd

from os.path import dirname, join


__all__ = ['load_trees']


def load_trees():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data', 'tree_parasite')
    Y = np.loadtxt(join(file_path, 'tree_adj.npy'))
    
    n_nodes = Y.shape[0]
    n_features = 3
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    X = np.zeros((n_dyads, n_features))
    
    # genetic distance
    Z = pd.read_csv(join(file_path, 'genetic_distances_ordered.csv'), 
            index_col=0).values
    X[:, 0] = np.log(Z[np.tril_indices_from(Z, k=-1)])
    
    # taxonomic distance
    Z = pd.read_csv(join(file_path, 'taxonomic_distance.csv'), delimiter=';').values
    X[:, 1] = Z[np.tril_indices_from(Z, k=-1)]

    # geographic distance
    Z = pd.read_csv(join(file_path, 'geographic_distance.csv'), delimiter=';').values
    X[:, 2] = Z[np.tril_indices_from(Z, k=-1)]

    return Y, X

