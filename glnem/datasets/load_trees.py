import numpy as np
import pandas as pd

from os.path import dirname, join


__all__ = ['load_trees']


def load_trees():
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
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    X = np.zeros((n_dyads, n_features))
    
    # NOTE: The previous analyses found taxonomic and geographic distance had a 
    #       negative effect on the number shared species, while genetic distance
    #       had no effect (although these variables are somewhat correlated,
    #       so collinearity could be masking an effect.
    
    # taxonomic distance
    Z = pd.read_csv(join(file_path, 'taxonomic_distance.csv'), delimiter=';').values
    X[:, 0] = Z[np.tril_indices_from(Z, k=-1)]
    
    # geographic distance
    Z = pd.read_csv(join(file_path, 'geographic_distance.csv'), delimiter=';').values
    X[:, 1] = Z[np.tril_indices_from(Z, k=-1)]
    
    # genetic distance
    Z = pd.read_csv(join(file_path, 'genetic_distances_ordered.csv'), 
            index_col=0).values
    X[:, 2] = np.log(Z[np.tril_indices_from(Z, k=-1)])

    return Y, X

