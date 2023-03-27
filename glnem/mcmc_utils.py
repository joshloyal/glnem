import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from jax import jit, random
from jax.scipy.special import expit
from scipy.linalg import orthogonal_procrustes


@jit
def polar_decomposition(X):
    eigvals, eigvecs = jnp.linalg.eigh(X.T @ X)
    XtX_sqrt_inv =  (eigvecs * (1. / jnp.sqrt(eigvals))) @ eigvecs.T
    return X @ XtX_sqrt_inv


@jit
def centered_qr_decomposition(X):
    n = X.shape[0]
    Q, R = jnp.linalg.qr(jnp.hstack((jnp.ones((n, 1)), X)))
    S = jnp.sign(jnp.diag(R))
    Q *= S
    return Q[:, 1:]


@jit
def ordered_expit(predictor, cutpoints):
    predictor = predictor[..., None]  # shape (n, 1)
    cumulative_probs = expit(cutpoints - predictor)
    pad_width = [(0, 0)] * (jnp.ndim(cumulative_probs) - 1) + [(1,1)]
    cumulative_probs = jnp.pad(cumulative_probs, pad_width, constant_values=(0, 1))
    return cumulative_probs[..., 1:] - cumulative_probs[..., :-1]


@jit
def stick_breaking(v):
    def break_stick(cumprod, v_i):
        w = v_i * cumprod
        return cumprod * (1 - v_i), w

    _, weights = jax.lax.scan(break_stick, jnp.array(1.0), v)

    return weights


@jit
def stick_breaking2(v):
    def break_stick(cumprod, v_i):
        w = v_i * cumprod
        return cumprod * (1 - v_i), w

    vs = jnp.hstack((v, 1.0))
    _, weights = jax.lax.scan(break_stick, jnp.array(1.0), vs)

    return weights


@jit
def cumulative_stick_breaking(v):
    return 1 - jnp.cumsum(stick_breaking(v))


def prior_effective_dimension(alpha, n_features):
    a = alpha / (1. + alpha)
    return np.sum(a ** np.arange(1, n_features + 1))


@jit
def Phi(t):
    return tfp.bijectors.NormalCDF().forward(t)


def flatten_array(X):
    return X.reshape(np.prod(X.shape[:-1]), -1)


def static_procrustes_rotation(X, Y):
    """Rotate Y to match X"""
    R, _ = orthogonal_procrustes(Y, X)
    return np.dot(Y, R)


def longitudinal_procrustes_rotation(X_ref, X):
    """A single procrustes transformation applied across time."""
    n_time_steps, n_nodes = X.shape[:-1]

    X_ref = flatten_array(X_ref)
    X = flatten_array(X)
    X = static_procrustes_rotation(X_ref, X)
    return X.reshape(n_time_steps, n_nodes, -1)
