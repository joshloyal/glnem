import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist

from jax.random import PRNGKey
from sklearn.utils import check_random_state

from ..glnem import LINK_FUNCS
from ..diagnostics import get_distribution
from ..tweedie import Tweedie
from ..network_utils import vec_to_adjacency


__all__ = ['synthetic_network']


def centered_qr_decomposition(X):
    n = X.shape[0]
    Q, R = np.linalg.qr(np.hstack((np.ones((n, 1)), X)))
    S = np.sign(np.diag(R))
    Q *= S
    return Q[:, 1:]


def centered_stiefel(n, p, random_state=None):
    rng = check_random_state(random_state)
    X = rng.randn(n, p)
    return centered_qr_decomposition(X)


def mixture_latent_space(n_nodes, n_features=3, random_state=123):
    rng = check_random_state(random_state)
    
    c = np.ones(n_features) / np.sqrt(n_features)
    mu = np.vstack((c, -c))
    z = rng.choice([0, 1], size=n_nodes)

    X = mu[z] + 0.1 * rng.randn(n_nodes, n_features)

    return centered_qr_decomposition(X), z


def generate_systematic_component(
        n_nodes=100, n_features=3, n_covariates=2, intercept=1.,
        family='bernoulli', random_state=123):
    rng = check_random_state(random_state)
     
    U, c = mixture_latent_space(n_nodes, n_features, rng)
    
    # eigenvalues
    z = rng.choice([-n_nodes, n_nodes], size=U.shape[1])
    if family in ['poisson', 'negbinom', 'tweedie']:
        lmbda = 0.5 * z  + np.sqrt(n_nodes) * rng.randn(U.shape[1])
    else:
        lmbda = z  + np.sqrt(n_nodes) * rng.randn(U.shape[1])
    
    # covariates
    if n_covariates is not None and n_covariates > 1:
        n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
        coefs = np.zeros(n_covariates)
        coefs[:2] = np.array([-0.5, 0.5])
        X = rng.uniform(-1, 1, size=(n_dyads, n_covariates))
    else:
        X = None
        coefs = None
    
    subdiag = np.tril_indices(n_nodes, k=-1)

    ULUt = ((U * lmbda) @ U.T)[subdiag]
    eta = intercept + ULUt 
    if X is not None:
        eta += X @ coefs
    
    params = {
            'intercept': intercept,
            'lambda': lmbda,
            'U': U,
            'coefs': coefs,
            'z': c,
            'similarities': ULUt, 
            'linear_predictor': eta
    }
    
    return eta, X, params


def tweedie_network(n_nodes=100, n_features='mixture', intercept=1., 
                    var_power=1.2, dispersion=3, random_state=123): 

    eta, X, params = generate_systematic_component(
            n_nodes=n_nodes, n_features=n_features, n_covariates=n_covariates, 
            intercept=intercept, random_state=random_state)

    twd =  Tweedie(loc=np.exp(eta[subdiag]), var_power=var_power, dispersion=dispersion)
    
    rng_key = PRNGKey(random_state)
    y_vec = twd.sample(rng_key)
    
    return y_vec, X, params


def count_network(n_nodes=100, n_features=2, n_covariates=2, intercept=2.5, dispersion=None, random_state=123):
    
    rng = check_random_state(random_state)
    
    lmbda = jnp.sqrt(n_nodes) * rng.randn(n_features)
    U = centered_stiefel(n_nodes, n_features, rng) 
    eta = intercept + U @ (lmbda * U).T
    
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    coefs = jnp.array([-1, 1])
    X = rng.randn(n_dyads, n_covariates)
    
    subdiag = np.tril_indices(n_nodes, k=-1)
    mu = jnp.exp(X @ coefs + eta[subdiag])
    if dispersion is None:
        dis = dist.Poisson(rate=mu)
    else:
        dis = dist.NegativeBinomial2(mean=mu, concentration=dispersion)
 
    rng_key = PRNGKey(random_state)
    y_vec = dis.sample(rng_key)
    
    params = {
            'intercept': intercept,
            'lambda': lmbda,
            'U': U,
            'coefs': coefs,
    }
    return vec_to_adjacency(y_vec), X, params


def synthetic_network(n_nodes=100, n_features='mixture', n_covariates=2, intercept=1., 
                      family='bernoulli', link='logit', dispersion=None, 
                      var_power=1.2, zif_prob=0.1, random_state=123):
    
    eta, X, params = generate_systematic_component(
            n_nodes=n_nodes, n_features=n_features, n_covariates=n_covariates, 
            intercept=intercept, family=family, random_state=random_state)

    mu = LINK_FUNCS[link](eta)
    params['mu'] = mu
    dist = get_distribution(
            mu, dispersion=dispersion, var_power=var_power, family=family,
            zif_prob=zif_prob)

    rng_key = PRNGKey(random_state)
    y_vec = dist.sample(rng_key)

    return vec_to_adjacency(y_vec), X, params
