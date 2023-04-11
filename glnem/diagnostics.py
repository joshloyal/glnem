import numpy as np

from numpyro import distributions as dist
from scipy.stats.distributions import norm, poisson, nbinom
from sklearn.utils import check_random_state

from .tobit import Tobit
from .tweedie import Tweedie
from .tweedie_dist import tweedie


def quantile_residuals(y, y_hat, dispersion=0., var_power=1.5, family='poisson', random_state=42):
    """Randomized Quantile Residuals"""
    rng = check_random_state(random_state)

    if family == 'poisson':
        lower = poisson.cdf(y - 1, mu=y_hat)
        upper = poisson.cdf(y, mu=y_hat)
    elif family == 'negbinom':
        sigma_sq = y_hat + dispersion * (y_hat ** 2)
        p = y_hat / sigma_sq
        n = (y_hat ** 2) / (sigma_sq - y_hat)
        lower = nbinom.cdf(y - 1, n=n, p=p)
        upper = nbinom.cdf(y, n=n, p=p)
    elif family == 'tweedie':
        u = tweedie(p=var_power, mu=y_hat, phi=dispersion).cdf(y) 
        u[y == 0] = rng.uniform(0, u[y==0])
        return norm.ppf(u)
    elif family == 'tobit':
        # TODO: Is this correct?
        u = norm(loc=y_hat, scale=dispersion).cdf(y)
        n_zero = np.sum(y == 0)
        u[y == 0] = rng.uniform(0, u[y==0])
        return norm.ppf(u)

    return norm.ppf(rng.uniform(lower, upper))


def get_distribution(mu, dispersion=1., var_power=1.5, family='bernoulli', zif_proba=0.1):

    if family == 'bernoulli':
        dis = dist.Bernoulli(mu)
    elif family == 'tweedie':
        dis = Tweedie(mu, var_power=var_power, dispersion=dispersion)
    elif family == 'tobit':
        dis = Tobit(mu, dispersion=dispersion)
    elif family == 'poisson':
        dis = dist.Poisson(mu)
    elif family == 'zif_poisson':
        dis = dist.ZeroInflatedPoisson(gate=zif_proba, rate=mu)
    elif family == 'negbinom':
        dis = dist.NegativeBinomial2(mean=mu, concentration=1/dispersion)
    elif family == 'normal':
        dis = dist.Normal(mu, dispersion)
    else:
        raise ValueError()
    
    return dis
