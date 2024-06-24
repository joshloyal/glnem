import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from joblib import Parallel, delayed
from jax.scipy.special import expit, logsumexp
from numpyro import handlers
from numpyro.contrib.control_flow import scan
from numpyro.distributions import constraints
from numpyro.distributions.transforms import OrderedTransform
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, MixedHMC, HMC, HMCGibbs
from numpyro.infer.util import log_density
from numpyro.infer import init_to_value, init_to_uniform, init_to_feasible, init_to_median
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.diagnostics import print_summary as numpyro_print_summary
from scipy.linalg import orthogonal_procrustes

from jax import jit, random, vmap
from jax.nn import relu
from sklearn.utils import check_random_state

from .tweedie import Tweedie, Chi2ZeroDoF
from .tobit import Tobit
from .mcmc_utils import (Phi, polar_decomposition, centered_qr_decomposition, 
        centered_projected_normal)
from .mcmc_utils import condition
from .network_utils import adjacency_to_vec
from .plots import plot_lpm
from .diagnostics import get_distribution


__all__ = ['LatentPositionModel']



def pairwise_distance(U):
    U_norm_sq = jnp.sum(U ** 2, axis=1).reshape(-1, 1)
    dist_sq = U_norm_sq + U_norm_sq.T - 2 * U @ U.T
    return dist_sq


def find_rotation(U, U_ref):
    R, _ = orthogonal_procrustes(U, U_ref)
    return U @ R

def posterior_predictive(model, rng_key, samples, stat_fun, *model_args,
                         **model_kwargs):
    #model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return stat_fun(model_trace["Y"]["value"])


def predict(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["mean"]["value"]


def linear_predictor(model, rng_key, samples, *model_args, **model_kwargs):
    #model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["linear_predictor"]["value"]


def similarities(model, rng_key, samples, *model_args, **model_kwargs):
    #model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["similarities"]["value"]


def predict_zero_probas(model, rng_key, samples, *model_args, **model_kwargs):
    #model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["zero_probas"]["value"]


def log_likelihood(model, rng_key, samples, *model_args, **model_kwargs):
    #model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model = handlers.seed(condition(model, samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    obs_node = model_trace["Y"]
    return obs_node["fn"].log_prob(obs_node["value"])


def print_summary(glnem, samples, feature_names, divergences, prob=0.9):
    # information criteria (large number of nodes makes this computation slow)
    if samples['U'].shape[1] < 150:
        waic = glnem.waic()
        print(f"WAIC: {waic:.3f}")
    
    fields = feature_names + ['intercept', 'lambda', 'sigma', 'dispersion', 'psi']
    if glnem.family == 'tweedie':
        fields = fields + ['var_power']

    samples = {k: v for k, v in samples.items() if
        k in fields and k in samples.keys()}
    samples = jax.tree_map(lambda x : jnp.expand_dims(x, axis=0), samples)
    
    # posterior summaries
    numpyro_print_summary(samples, prob=prob)
    if divergences is not None:
        print(f"Number of divergences: {divergences}")


def calculate_zero_probas(mu, family, dispersion, var_power):
    if family == 'poisson':
        return jnp.exp(-mu)
    elif family == 'negbinom':
        return (dispersion / (mu + dispersion)) ** dispersion 
    elif family == 'tweedie':
        return jnp.exp(-(mu ** (2 - var_power))/ (dispersion * (2 - var_power)))
    elif family == 'tobit':
        return 1 - Phi(mu / dispersion)
    else:
        return mu


def cloglog(x):
    return 1 - jnp.exp(-jnp.exp(eta))


def loglog(x):
    return jnp.exp(-jnp.exp(eta))


def identity(x):
    return x


LINK_FUNCS = {
        'identity': identity,
        'logit': expit,
        'log': jnp.exp,
        'softplus': jax.nn.softplus,
        'probit': Phi,
        'cloglog': cloglog,
        'loglog': loglog,
        'sqrt': jnp.sqrt
}


def lpm(Y, Z, n_nodes, train_indices,
        n_features=2, family='bernoulli', link='identity', ls_type='distance',
        is_predictive=False):

    sigma = numpyro.sample('sigma', dist.InverseGamma(1.5, 1.5)) # 1 * invX2(1)

    # latent space
    X = numpyro.sample("X",
        dist.Normal(jnp.zeros((n_nodes, n_features)),
                    jnp.ones((n_nodes, n_features))))
    U = numpyro.deterministic("U", jnp.sqrt(sigma) * X)

    # intercept
    intercept = numpyro.sample('intercept', dist.Normal(0, 10))

    if Z is not None:
        n_covariates = Z.shape[1]
        coefs = numpyro.sample('coefs',
            dist.Normal(jnp.zeros(n_covariates), jnp.repeat(10., n_covariates)))

    if family in ['gaussian', 'negbinom', 'zif_negbinom', 'tweedie', 'tobit']:
        dispersion = numpyro.sample("dispersion", dist.HalfCauchy(scale=1))
    else:
        dispersion = None

    if family in ['negbinom', 'zif_negbinom']:
        psi = numpyro.deterministic('psi', 1 / dispersion)

    if family in ['zif_poisson', 'zif_negbinom']:
        q = numpyro.sample('q', dist.Uniform(low=0, high=1))
    
    if family == 'tweedie':
        # XXX: We restrict p to (1.01, 1.99) because the distribution becomes
        #      very multimodal as it converges to a discrete (p = 1, poisson) and 
        #      continous (p = 2, gamma) distribution.
        var_power = numpyro.sample("var_power", dist.Uniform(low=1.01, high=1.99))
    else:
        var_power = numpyro.deterministic('var_power', 0.0)

    # euclidean distance predictor
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    if ls_type == 'distance':
        dis = pairwise_distance(U)
        eta = intercept - jnp.sqrt(dis[subdiag])
    elif ls_type == 'distance_sq':
        dis = pairwise_distance(U)
        eta = intercept - dis[subdiag] 
    else:
        eta = intercept + (U @ U.T)[subdiag]

    if Z is not None:
        eta += Z @ coefs

    # likelihood
    with numpyro.handlers.condition(data={"Y": Y}):
        with numpyro.handlers.mask(mask=train_indices):
            
            mu = (eta if (family == 'bernoulli' and link == 'logit') else 
                    LINK_FUNCS[link](eta))

            if family == 'bernoulli':
                if link == 'logit':
                    y = numpyro.sample("Y", dist.Bernoulli(logits=mu))
                else:
                    y = numpyro.sample("Y", dist.Bernoulli(mu))
            elif family == 'tweedie':
                y = numpyro.sample("Y", 
                        Tweedie(mu, var_power=var_power, dispersion=dispersion)) 
            elif family == 'tobit':
                y = numpyro.sample("Y", Tobit(mu, dispersion=dispersion))
            elif family == 'poisson':
                y = numpyro.sample("Y", dist.Poisson(mu))
            elif family == 'zif_poisson':
                y = numpyro.sample("Y", dist.ZeroInflatedPoisson(gate=q, rate=mu))
            elif family == 'negbinom':
                y = numpyro.sample("Y", dist.NegativeBinomial2(mean=mu, concentration=psi))
            elif family == 'zif_negbinom':
                y = numpyro.sample("Y", dist.ZeroInflatedNegativeBinomial2(mean=mu, concentration=psi, gate=q))
            else:
                y = numpyro.sample("Y", dist.Normal(mu, dispersion))

    if is_predictive:
        mu = LINK_FUNCS[link](eta)
        numpyro.deterministic("similarities", dist)
        numpyro.deterministic("linear_predictor", eta)
        numpyro.deterministic("mean", mu)
        numpyro.deterministic("zero_probas", 
                calculate_zero_probas(mu, family, dispersion, var_power))


class LatentPositionModel(object):
    """Latent Position Model"""
    def __init__(self,
                 n_features=2,
                 family='bernoulli',
                 link='logit',
                 ls_type='distance',
                 random_state=42):
        self.n_features = n_features
        self.family = family
        self.link = link
        self.ls_type = ls_type
        self.random_state = random_state
    
    @property
    def model_args_(self):
        n_nodes = self.samples_['U'].shape[1]
        return (None, self.X_dyad_, n_nodes, True, self.n_features,  
                self.family, self.link, self.ls_type)

    @property
    def model_kwargs_(self):
        return {'is_predictive': True}
    
    @property
    def n_params_(self):
        """
        # of U params, Lambda Params, Intercept, and Coefficients.
        """
        n_covariates = self.coefs_.shape[0] if self.coefs_ is not None else 0
        return (
                np.prod(self.U_.shape) +    # latent space
                n_covariates + 1    # covariate-effects + intercept
        )

    def sample(self, Y, X=None, n_warmup=1000, n_samples=1000, 
               missing_edges=None, adapt_delta=0.8,
               thinning=1):
        numpyro.enable_x64()

        # network to dyad list
        n_nodes = Y.shape[0]
        y = adjacency_to_vec(Y)
        self.y_fit_ = y
        #train_indices = y != -1
        #self.train_indices_ = train_indices

        if missing_edges is not None:
            self.train_indices_ = np.repeat(True, y.shape[0])
            self.train_indices_[missing_edges] = False
        else:
            self.train_indices_ = True
        
        if X is None:
            self.X_dyad_ = None
            self.feature_names_ = []
        else:
            if X.ndim != 2 or X.shape[0] != y.shape[0]:
                raise ValueError("X has the wrong shape.")
            self.feature_names_ = [
                'X{}'.format(p+1) for p in range(X.shape[-1])]
            self.X_dyad_ = jnp.asarray(X)

        # run mcmc sampler
        rng_key = random.PRNGKey(self.random_state)
        model_args = (
            y, self.X_dyad_, n_nodes, self.train_indices_, self.n_features, 
            self.family, self.link, self.ls_type)
        model_kwargs = {'is_predictive': False}
        init_values = init_to_uniform()

        kernel = NUTS(lpm, target_accept_prob=adapt_delta,
                      init_strategy=init_values)
        
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                    thinning=thinning, num_chains=1)
        mcmc.run(rng_key, *model_args, **model_kwargs)
        self.diverging_ = None

        # extract/process samples
        self.samples_ = mcmc.get_samples()

        # calculate log density
        self.logp_ = vmap(
            lambda sample : log_density(
                lpm, model_args=model_args, model_kwargs=model_kwargs,
                params=sample)[0])(self.samples_)
        self.map_idx_ = np.argmax(self.logp_)

        self.samples_ = jax.tree_map(lambda x : np.array(x), self.samples_)
        
        # save covariate effects
        if X is not None:
            coefs = self.samples_['coefs']
            for p, name in enumerate(self.feature_names_):
                self.samples_[name] = coefs[:, p]
            self.coefs_ = coefs.mean(axis=0)
        else:
            self.coefs_ = None

        # save dispersion
        if self.family in ['gaussian', 'negbinom', 'zif_negbinom', 'tweedie', 'tobit']:
            self.dispersion_ = self.samples_['dispersion'].mean()
        else:
            self.dispersion_ = None

        # tweedie var power
        if self.family == 'tweedie':
            self.var_power_ = self.samples_['var_power'].mean() 
        else:
            self.var_power_ = None

        # fix permutation issue for each sample
        U_ref = self.samples_['U'][self.map_idx_]
        for idx in range(self.samples_['U'].shape[0]):
            self.samples_['U'][idx] = find_rotation(
                self.samples_['U'][idx], U_ref)

        # posterior means
        self.sigma_ = self.samples_['sigma'].mean(axis=0)
        self.intercept_ = self.samples_['intercept'].mean(axis=0)
        self.U_ = self.samples_['U'].mean(axis=0)

        return self

    def waic(self, Y=None, X=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        n_nodes = self.samples_['U'].shape[1]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        
        y = adjacency_to_vec(Y) if Y is not None else self.y_fit_
        train_indices = True if Y is not None else self.train_indices_
        X_dyad = jnp.asarray(X) if X is not None else self.X_dyad_

        model_args = (y, X_dyad, n_nodes, train_indices, self.n_features, 
                self.family, self.link)
        
        loglik = vmap(
            lambda samples, rng_key : log_likelihood(
                lpm, rng_key, samples,
                *model_args, **self.model_kwargs_))(*vmap_args)

        lppd = (logsumexp(loglik, axis=0) - jnp.log(n_samples)).sum()
        p_waic = loglik.var(axis=0).sum()
        return float(-2 * (lppd - p_waic))

    def dic(self, Y=None, X=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        n_nodes = self.samples_['U'].shape[1]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        y = adjacency_to_vec(Y) if Y is not None else self.y_fit_
        train_indices = True if Y is not None else self.train_indices_
        X_dyad = jnp.asarray(X) if X is not None else self.X_dyad_
        
        model_args = (y, X_dyad, n_nodes, train_indices, self.n_features, 
                self.family, self.link)

        loglik = vmap(
            lambda samples, rng_key : log_likelihood(
                lpm, rng_key, samples,
                *model_args, **self.model_kwargs_))(*vmap_args)

        loglik_hat = self.loglikelihood()
        p_dic = 2 * (loglik_hat - loglik.sum(axis=1).mean()).item()

        return -2 * loglik_hat + 2 * p_dic

    def bic(self):
        n_nodes = self.U_.shape[0]
        n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
        
        loglik_hat = self.loglikelihood()
        return -2 * loglik_hat + np.log(n_dyads) * self.n_params_

    def aic(self):
        n_nodes = self.U_.shape[0]
         
        loglik_hat = self.loglikelihood()
        return -2 * loglik_hat + 2 * self.n_params_ 

    def posterior_predictive(self, stat_fun=None, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        
        if stat_fun is None:
            # posterior predictive draws of the network itself
            stat_fun = lambda x: x

        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                lpm, rng_key, samples, stat_fun,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args))
    
    def loglikelihood(self, Y=None, test_indices=None):
        y = adjacency_to_vec(Y) if Y is not None else self.y_fit_
        X = self.X_dyad_

        n_dyads = y.shape[0]
        n_nodes = self.U_.shape[0]

        subdiag = np.tril_indices(n_nodes, k=-1)
        dis = pairwise_distances(U)
        eta = self.intercept_ - np.sqrt(dis[subdiag]) 
        if X is not None:
            eta += X @ self.coefs_

        if test_indices is not None:
            eta = eta[test_indices]
            y_true = y[test_indices]
        else:
            y_true = y
        
        mu = LINK_FUNCS[self.link](eta)
        dis = get_distribution(mu, 
                dispersion=self.dispersion_, var_power=self.var_power_,
                family=self.family)
        loglik = dis.log_prob(y_true).sum()

        return np.asarray(loglik).item()
    
    def predict(self, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : predict(
                lpm, rng_key, samples,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0))
    
    def linear_predictor(self, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : linear_predictor(
                lpm, rng_key, samples,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0))
    
    def similarities(self, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : similarities(
                lpm, rng_key, samples,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0))
    
    def predict_zero_probas(self, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : predict_zero_probas(
                lpm, rng_key, samples,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0))

    def print_summary(self, proba=0.9):
        print_summary(self,
                self.samples_, self.feature_names_, self.diverging_, prob=proba)

    def plot(self, Y_obs=None, include_diagnostics=True, **fig_kwargs):
        return plot_lpm(self, Y_obs, include_diagnostics, **fig_kwargs)
