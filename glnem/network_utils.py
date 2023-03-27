import jax
import jax.numpy as jnp
import numpy as np


def shape_from_tril_vec(x, k=0):
    return round((np.sqrt(1 + 8 * x.shape[-1]) - 1) / 2) - k


def vec_to_adjacency(y_vec, include_nan=False):
    n = shape_from_tril_vec(y_vec, k=-1)
    indices = np.tril_indices(n, k=-1)
    Y = jnp.zeros((n, n))
    Y = Y.at[indices].set(y_vec)
    Y += Y.T
    if include_nan:
        Y.at[jnp.diag_indices(n)].set(jnp.nan)
    return Y


def adjacency_to_vec(Y):
    n_nodes = Y.shape[0]
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    return Y[subdiag]


def dynamic_adjacency_to_vec(Y):
    n_time_steps, n_nodes, _ = Y.shape
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    subdiag = np.tril_indices(n_nodes, k=-1)
    y = jnp.zeros((n_time_steps, n_dyads), dtype=np.int64)
    for t in range(n_time_steps):
        y = y.at[t].set(Y[t][subdiag].astype('int32'))

    return y
