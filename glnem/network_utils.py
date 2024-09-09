import jax
import jax.numpy as jnp
import numpy as np


def shape_from_tril_vec(x, k=0):
    return round((np.sqrt(1 + 8 * x.shape[-1]) - 1) / 2) - k


def vec_to_adjacency(y_vec, include_diag=False):
    k = 0 if include_diag else -1
    n = shape_from_tril_vec(y_vec, k=k)
    indices = np.tril_indices(n, k=k)
    Y = np.zeros((n, n))
    Y[indices] = y_vec
    Y += Y.T
    Y[np.diag_indices_from(Y)] *= 0.5
    return Y


def adjacency_to_vec(Y, include_diag=False):
    n_nodes = Y.shape[0]
    if include_diag:
        indices = np.tril_indices(n_nodes)
    else:
        indices = np.tril_indices(n_nodes, k=-1)
    return Y[indices]


def dynamic_adjacency_to_vec(Y):
    n_time_steps, n_nodes, _ = Y.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    y = jnp.zeros((n_time_steps, n_dyads), dtype=np.int64)
    for t in range(n_time_steps):
        y = y.at[t].set(Y[t][subdiag].astype('int32'))

    return y


def node_averaged_rank(X_dyad):
    n_dyads, n_covariates = X_dyad.shape
    n_nodes = int(-0.5 + 0.5 * np.sqrt(1 + 8 * n_dyads))
    X = np.zeros((n_nodes, n_nodes, n_covariates))
    for k in range(n_covariates):
        X[..., k] = vec_to_adjacency(X_dyad[..., k], include_diag=True)
    
    # calculate node-averaged covariates
    Xbar = X.mean(axis=0)
    Xbar = np.c_[np.ones(n_nodes), Xbar]  # add intercept column

    return np.linalg.matrix_rank(Xbar)
