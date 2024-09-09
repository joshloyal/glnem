import numpy as np
import pandas as pd

from os.path import dirname, join


__all__ = ['load_trees']


def load_trees(include_diag=False):
    """Number of parasites shared by tree species. Nodes are tree species and
    edge weights are the number of shared parasitic species between trees.
    
    References
    ----------
        * Accelerating Bayesian Estimation for Network Poisson Models 
            Using Frequentist Variational Estimates.
        * Uncovering Latent Structure in Valued Graphs: A Variational Approach.
    """
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data', 'tree_parasite')
    Y = np.loadtxt(join(file_path, 'tree_adj.npy'))
    
    n_nodes = Y.shape[0]
    n_features = 3

    if include_diag:
        n_dyads = int(0.5 * n_nodes * (n_nodes + 1))
    else:
        n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    X = np.zeros((n_dyads, n_features))
    
    # NOTE: The previous analyses found taxonomic and geographic distance had a 
    #       negative effect on the number shared species, while genetic distance
    #       had no effect (although these variables are somewhat correlated,
    #       so collinearity could be masking an effect.
    
    k = 0 if include_diag else -1

    # taxonomic distance
    Z = pd.read_csv(join(file_path, 'taxonomic_distance.csv'), delimiter=';').values
    X[:, 0] = Z[np.tril_indices_from(Z, k=k)]
    
    # geographic distance
    Z = pd.read_csv(join(file_path, 'geographic_distance.csv'), delimiter=';').values
    X[:, 1] = Z[np.tril_indices_from(Z, k=k)]
    
    # genetic distance
    Z = pd.read_csv(join(file_path, 'genetic_distances_ordered.csv'), 
            index_col=0).values
    #X[:, 2] = np.log(Z[np.tril_indices_from(Z, k=k)])
    X[:, 2] = Z[np.tril_indices_from(Z, k=k)]
    if include_diag:
        nonzero_mask = X[:, 2] > 0
        X[:, 2][nonzero_mask] = np.log(X[:, 2][nonzero_mask])
    else:
        X[:, 2] = np.log(X[:, 2])

    return Y, X

