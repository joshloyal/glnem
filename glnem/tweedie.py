from jax import lax
from numpyro.distributions.distribution import Distribution
from numpyro.distributions import constraints
from numpyro.distributions.util import is_prng_key, promote_shapes
from numpyro.distributions import Gamma, Poisson
from jax.lax import lgamma
from jax.scipy.special import logsumexp
from jax.lax import bessel_i1e, while_loop

import jax.numpy as jnp
import jax.random as random


#EPS = 37
EPS = 17


class Tweedie(Distribution):
    arg_constraints = {
        "loc": constraints.positive,
        "dispersion": constraints.positive,
        "var_power": constraints.interval(1, 2),
    }
    reparametrized_params = ["loc", "dispersion", "var_power"]
    support = constraints.real
    def __init__(self, loc=1.0, dispersion=1.0, var_power=1.5, *, validate_args=None): 
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(dispersion), jnp.shape(var_power)
        )
        self.loc, self.dispersion, self.var_power = promote_shapes(
            loc, dispersion, var_power, shape=batch_shape
        )

        super(Tweedie, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def log_prob(self, value):
        return tweedie_logp(value, self.loc, self.var_power, self.dispersion)

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        key_poisson, key_gamma = random.split(key)
        
        rate = (self.loc ** (2 - self.var_power)) / (self.dispersion * (2 - self.var_power))
        m = Poisson(rate).sample(key_poisson, sample_shape)
        
        shape = -(2 - self.var_power) / (1 - self.var_power)
        scale = self.dispersion * (self.var_power - 1) * jnp.power(self.loc, self.var_power - 1)
        return jnp.where(
                m > 0.,
                Gamma(m * shape, rate=1./scale).sample(key_gamma),
                0.)
    

class Chi2ZeroDoF(Tweedie):
    """Tweedie distribution with var_power = 1.5"""
    def __init__(self, loc=1.0, dispersion=1.0, *, validate_args=None):
        super(Chi2ZeroDoF, self).__init__(
                loc=loc, dispersion=dispersion, var_power=1.5, validate_args=validate_args)

    def log_prob(self, value):
        p = self.var_power
        theta = (self.loc ** (1 - p)) / (1 - p)
        ktheta = (self.loc ** (2 - p)) / (2 - p)
        dens = (value * theta - ktheta) / self.dispersion

        z = jnp.sqrt(value) / self.dispersion
        return dens + z + jnp.log(jnp.where(
                value > 0.,
                bessel_i1e(z),
                1.))


def tweedie_logp(y, mu, p, tau):
    return jnp.where(
            y > 0.,
            tweedie_logp_series(jnp.where(y > 0, y, 1.), mu, p, tau),
            -(mu ** (2 - p) / (tau * (2 - p))),
        )

def tweedie_logp_series(y, mu, p, tau, max_j=25.):
    #j_range = lax.stop_gradient(get_jrange(y, p, tau))
    #j_range = jnp.arange(1., 100.)[:, jnp.newaxis]
    j_range = jnp.arange(1., max_j)[:, jnp.newaxis]
    log_w = logsumexp(log_wj(y, p, tau, j_range), axis=0)
    
    theta = (mu ** (1 - p)) / (1 - p)
    ktheta = (mu ** (2 - p)) / (2 - p)
    return log_w - jnp.log(y) + ((y * theta - ktheta) / tau)


def log_z(y, p, tau):
    alpha = (2 - p) / (1 - p)
    return (
        alpha * jnp.log(p - 1) -
        alpha * jnp.log(y) -
        (1 - alpha) * jnp.log(tau) -
        jnp.log(2 - p)
    )  


def log_wj(y, p, tau, j):
    alpha = (2 - p) / (1 - p)
    return j * log_z(y, p, tau) - lgamma(1 + j) - lgamma(-alpha * j)


def log_wmax(j_max, p):
    alpha = (2 - p) / (1 - p)
    return j_max * (alpha - 1) - jnp.log(2 * jnp.pi) - jnp.log(j_max) - 0.5 * jnp.log(-alpha)


def get_jrange(y, p, tau):
    j_max = (y ** (2 - p)) / ((2 - p) * tau)

    # start at maximum over all y values
    j_hi = jnp.maximum(1, jnp.max(j_max))
    logWmax = log_wmax(j_hi, p)
    
    # increase j away from j_max until Wj is beyoned e-37 sig figs
    def cond_fun(val):
        res = log_wj(y, p, tau, val) 
        return jnp.any((logWmax - res) < EPS) 

    def body_fun(val):
        return val + 1
    
    j = while_loop(cond_fun, body_fun, j_hi)
    j_hi = jnp.ceil(j)

    # start at minimum over all y values
    j_low = jnp.maximum(1, jnp.min(j_max))
    logWmax = log_wmax(j_low, p)
    
    # decrease j away from j_max until Wj is beyoned e-37 sig figs
    def cond_fun(val):
        res = log_wj(y, p, tau, val) 
        return jnp.logical_and(
                jnp.any(((logWmax - res) < EPS)),
                jnp.all(val > 0)
        )
    
    def body_fun(val):
        return val - 1
    
    j = while_loop(cond_fun, body_fun, j_low)
    j_low = jnp.ceil(j)
    
    
    return jnp.arange(j_low, j_hi + 1)
