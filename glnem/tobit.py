from jax import lax

from numpyro.distributions.distribution import Distribution
from numpyro.distributions import constraints, Normal
from numpyro.distributions.util import is_prng_key, promote_shapes

import jax.numpy as jnp
import jax.random as random

from .mcmc_utils import Phi


class Tobit(Distribution):
    arg_constraints = {
        "loc": constraints.positive,
        "dispersion": constraints.positive,
    }
    reparametrized_params = ["loc", "dispersion"]
    support = constraints.real
    def __init__(self, loc=1.0, dispersion=1.0, *, validate_args=None): 
        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc), jnp.shape(dispersion)
        )
        self.loc, self.dispersion = promote_shapes(
            loc, dispersion, shape=batch_shape
        )
        
        self.cont_dist = Normal(loc, self.dispersion)
        super(Tobit, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def log_prob(self, value):
        return jnp.where(value > 0.,
                self.cont_dist.log_prob(value),
                jnp.log(1 - Phi(self.loc / self.dispersion))
        )

    def sample(self, key, sample_shape=()):
        assert is_prng_key(key)
        
        return jnp.maximum(self.cont_dist.sample(key, sample_shape), 0)
