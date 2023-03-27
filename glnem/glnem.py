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
from numpyro.infer import init_to_value, init_to_uniform, init_to_feasible, init_to_median
from numpyro.infer import SVI, Trace_ELBO, autoguide
from numpyro.diagnostics import print_summary as numpyro_print_summary
from scipy.optimize import linear_sum_assignment

from jax import jit, random, vmap
from jax.nn import relu
from sklearn.utils import check_random_state

from .tweedie import Tweedie, Chi2ZeroDoF
from .tobit import Tobit
from .mcmc_utils import Phi, polar_decomposition, centered_qr_decomposition
from .network_utils import adjacency_to_vec
from .plots import plot_glnem
from .diagnostics import get_distribution


__all__ = ['GLNEM']


def find_permutation(U, U_ref):
    n_features = U.shape[1]
    C = U_ref.T @ U
    perm = linear_sum_assignment(np.maximum(C, -C), maximize=True)[1]
    sign = np.sign(C[np.arange(n_features), perm])
    return sign * U[..., perm], perm


def posterior_predictive(model, rng_key, samples, stat_fun, *model_args,
                         **model_kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return stat_fun(model_trace["Y"]["value"])


def predict(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["mean"]["value"]


def predict_zero_probas(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    return model_trace["zero_probas"]["value"]


def log_likelihood(model, rng_key, samples, *model_args, **model_kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    model_trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
    obs_node = model_trace["Y"]
    return obs_node["fn"].log_prob(obs_node["value"])



def calculate_posterior_predictive(mcmc, stat_fun, random_state, *model_args,
                                   **model_kwargs):
    rng_key = random.PRNGKey(random_state)

    samples = mcmc.get_samples()
    n_samples  = samples['U'].shape[0]
    vmap_args = (samples, random.split(rng_key, n_samples))

    return vmap(
        lambda samples, rng_key : posterior_predictive(
            model, rng_key, samples, stat_fun,
            *model_args, **model_kwargs)
    )(*vmap_args)


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


def glnem(Y, Z, n_nodes, train_indices,
          n_features=2, v0=0., family='bernoulli', link='identity',
          tweedie_var_power=None, infer_dimension=False,
          infer_sigma=True, is_predictive=False):

    # sparsity factor
    eps = 0.1
    if infer_sigma:
        sigmak = numpyro.sample("sigmak",
            dist.Exponential(rate=1/n_nodes).expand([n_features]))
    else:
        sigmak = jnp.repeat(n_nodes ** 2, n_features)

    if infer_dimension:
        # Spike-and-Slab IBP
        v1 = numpyro.sample("v1", dist.Beta(
            (1/n_features), 1 + n_features ** (1 + eps)))
        vh = numpyro.sample("vh", dist.Beta(
            (1/n_features) * jnp.ones(n_features-1), jnp.ones(n_features - 1)))
        v = numpyro.deterministic("v", jnp.concatenate((jnp.array([v1]), vh)))
        w = numpyro.deterministic("w", jnp.cumprod(v))
        s = numpyro.sample("s", dist.Bernoulli(probs=w))
        z = numpyro.deterministic("z", s)
        thetak = jnp.where(s, 1, jnp.sqrt(v0))
    else:
        thetak = 1.
    
    sigma = numpyro.deterministic('sigma', thetak * jnp.sqrt(sigmak))

    # lambda values
    lmbda0 = numpyro.sample("lambda0", dist.Normal(0, 1).expand([n_features]))
    lmbda = numpyro.deterministic("lambda", sigma * lmbda0)

    # latent space
    X = numpyro.sample("X",
        dist.Normal(jnp.zeros((n_nodes, n_features)),
                    jnp.ones((n_nodes, n_features))))
    U = numpyro.deterministic("U", centered_qr_decomposition(X))

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
    
    if family == 'tweedie' and tweedie_var_power is None:
        # XXX: We restrict p to (1.01, 1.99) because the distribution becomes
        #      very multimodal as it converges to a discrete (p = 1, poisson) and 
        #      continous (p = 2, gamma) distribution.
        var_power = numpyro.sample("var_power", dist.Uniform(low=1.01, high=1.99))
    else:
        p = 0.0 if tweedie_var_power is None else tweedie_var_power
        var_power = numpyro.deterministic('var_power', p)

    # bilinear predictor
    subdiag = jnp.tril_indices(n_nodes, k=-1)
    eta = intercept + ((U * lmbda) @ U.T)[subdiag]

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
            numpyro.deterministic("linear_predictor", eta)
            numpyro.deterministic("mean", mu)
            numpyro.deterministic("zero_probas", 
                    calculate_zero_probas(mu, family, dispersion, var_power))


class GLNEM(object):
    """Generalized Linear Network Eigenmodel"""
    def __init__(self,
                 n_features=8,
                 family='bernoulli',
                 link='logit',
                 infer_dimension=True,
                 infer_sigma=True,
                 tweedie_var_power=None,
                 v0=0.,
                 random_state=42):
        self.n_features = n_features
        self.infer_dimension = infer_dimension
        self.infer_sigma = infer_sigma
        self.v0 = v0
        self.family = family
        self.link = link
        self.tweedie_var_power = tweedie_var_power
        self.random_state = random_state
    
    @property
    def model_args_(self):
        n_nodes = self.samples_['U'].shape[1]
        return (None, self.X_dyad_, n_nodes, True, self.n_features, self.v0, 
                self.family, self.link,
                self.tweedie_var_power)

    @property
    def model_kwargs_(self):
        return {'infer_dimension': self.infer_dimension,
                'infer_sigma': self.infer_sigma, 'is_predictive': True}

    def sample(self, Y, X=None, n_warmup=1000, n_samples=1000, adapt_delta=0.8,
               n_iter=10000, thinning=1, num_particles=1):
        numpyro.enable_x64()

        # network to dyad list
        n_nodes = Y.shape[0]
        y = adjacency_to_vec(Y)
        train_indices = y != -1
        self.y_fit_ = y
        self.train_indices_ = train_indices
        
        if X is None:
            self.X_dyad_ = None
            self.feature_names_ = []
        else:
            if X.ndim != 2 or X.shape[0] != y.shape[0]:
                raise ValueError()
            self.feature_names_ = [
                'X{}'.format(p+1) for p in range(X.shape[-1])]
            self.X_dyad_ = jnp.asarray(X)

        # run mcmc sampler
        rng_key = random.PRNGKey(self.random_state)
        model_args = (
            y, self.X_dyad_, n_nodes, train_indices, self.n_features, self.v0, 
            self.family, self.link, self.tweedie_var_power)
        model_kwargs = {
            'infer_dimension': self.infer_dimension,
            'infer_sigma': self.infer_sigma,
            'is_predictive': False}

        if self.infer_dimension:
            kernel = NUTS(glnem, target_accept_prob=adapt_delta)
            mcmc = MCMC(kernel, num_warmup=250, num_samples=250, num_chains=1)
            init_kwargs = model_kwargs.copy()
            init_kwargs['infer_dimension'] = False
            mcmc.run(rng_key, *model_args, **init_kwargs)
            init_values = init_to_value(values={
                key: value[-1] for key, value in mcmc.get_samples().items()
            })
        else:
            init_values = init_to_uniform()

        kernel = NUTS(glnem, target_accept_prob=adapt_delta,
                      init_strategy=init_values)
        
        if self.infer_dimension:
            kernel = DiscreteHMCGibbs(kernel, modified=False)

        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                    thinning=thinning, num_chains=1)
        mcmc.run(rng_key, *model_args, **model_kwargs)

        if not self.infer_dimension:
            self.diverging_ = jnp.sum(mcmc.get_extra_fields()['diverging'])
        else:
            self.diverging_ = None

        # extract/process samples
        self.samples_ = mcmc.get_samples()

        self.samples_ = jax.tree_map(lambda x : np.array(x), self.samples_)
        
        # save covariate effects
        if X is not None:
            coefs = self.samples_['coefs']
            for p, name in enumerate(self.feature_names_):
                self.samples_[name] = coefs[:, p]
            self.coefs_ = coefs.mean(axis=0)

        # save dispersion
        if self.family in ['gaussian', 'negbinom', 'zif_negbinom', 'tweedie', 'tobit']:
            self.dispersion_ = self.samples_['dispersion'].mean()
        else:
            self.dispersion_ = None

        # fix permutation issue for each sample
        U_ref = self.samples_['U'][-1]
        for idx in range(self.samples_['U'].shape[0]):
            self.samples_['U'][idx], perm = find_permutation(
                self.samples_['U'][idx], U_ref)
            self.samples_['lambda'][idx] = (
                self.samples_['lambda'][idx, perm].T)
            self.samples_['sigma'][idx] = (
                self.samples_['sigma'][idx, perm].T)

            if self.infer_dimension:
                self.samples_['s'][idx] = self.samples_['s'][idx, perm]
                self.samples_['z'][idx] = self.samples_['z'][idx, perm]

        if self.infer_dimension:
            # order columns based on inclusion probabilities and then by lmbda
            self.inclusion_probas_ = self.samples_['s'].mean(axis=0)
            imp_perm = np.lexsort(
                (self.samples_['lambda'].mean(axis=0),
                 self.inclusion_probas_))[::-1]
            self.inclusion_probas_ = self.inclusion_probas_[imp_perm]
            for idx in range(self.samples_['U'].shape[0]):
                self.samples_['U'][idx] = self.samples_['U'][idx][:, imp_perm]
                self.samples_['lambda'][idx] = (
                    self.samples_['lambda'][idx, imp_perm].T)
                self.samples_['sigma'][idx] = self.samples_['sigma'][idx, imp_perm]
                self.samples_['s'][idx] = self.samples_['s'][idx, imp_perm]
                self.samples_['z'][idx] = self.samples_['z'][idx, imp_perm]
        else:
            # order columns based on magnitude of lambda
            imp_perm = np.argsort(self.samples_['lambda'].mean(axis=0))[::-1]
            for idx in range(self.samples_['U'].shape[0]):
                self.samples_['U'][idx] = self.samples_['U'][idx][:, imp_perm]
                self.samples_['lambda'][idx] = (
                    self.samples_['lambda'][idx, imp_perm].T)
                self.samples_['sigma'][idx] = self.samples_['sigma'][idx, imp_perm]


        # posterior means
        self.sigma_ = self.samples_['sigma'].mean(axis=0)
        self.lambda_ = self.samples_['lambda'].mean(axis=0)
        self.intercept_ = self.samples_['intercept'].mean(axis=0)
        self.U_mean_ = self.samples_['U'].mean(axis=0)
        self.U_ = polar_decomposition(self.U_mean_)

        return self

    def waic(self, Y=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        n_nodes = self.samples_['U'].shape[1]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))
        
        y = adjacency_to_vec(Y) if Y is not None else self.y_fit_
        train_indices = True if Y is not None else self.train_indices_

        model_args = (y, self.X_dyad_, n_nodes, train_indices, self.n_features, self.v0, 
                self.family, self.link,
                self.tweedie_var_power)
        
        loglik = vmap(
            lambda samples, rng_key : log_likelihood(
                glnem, rng_key, samples,
                *model_args, **self.model_kwargs_))(*vmap_args)

        lppd = (logsumexp(loglik, axis=0) - jnp.log(n_samples)).sum()
        p_waic = loglik.var(axis=0).sum()
        return float(-2 * (lppd - p_waic))

    def dic(self, Y=None, random_state=0):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        n_nodes = self.samples_['U'].shape[1]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        y = adjacency_to_vec(Y) if Y is not None else self.y_fit_
        train_indices = True if Y is not None else self.train_indices_
        
        model_args = (y, self.X_dyad_, n_nodes, train_indices, self.n_features, self.v0, 
                self.family, self.link, self.tweedie_var_power)

        loglik = vmap(
            lambda samples, rng_key : log_likelihood(
                glnem, rng_key, samples,
                *self.model_args_, **self.model_kwargs_))(*vmap_args)

        loglik_hat = self.loglikelihood(y)
        p_dic = 2 * (loglik_hat - loglik.sum(axis=1).mean()).item()

        return -2 * (loglik_hat - p_dic)

    def bic(self, Y):
        n_nodes, _ = Y.shape
        n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
        loglik_hat = self.loglikelihood(adjacency_to_vec(Y))
        p_bic = np.log(n_dyads) * (n_nodes + 1) * self.n_features
        return -2 * loglik_hat + p_bic

    def aic(self, Y):
        n_nodes, _ = Y.shape
        loglik_hat = self.loglikelihood(adjacency_to_vec(Y))
        return -2 * loglik_hat + (n_nodes + 1) * self.n_features

    def posterior_predictive(self, stat_fun, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : posterior_predictive(
                glnem, rng_key, samples, stat_fun,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args))

    def loglikelihood(self, y, test_indices=None):
        n_dyads = y.shape[0]
        n_nodes = self.U_.shape[0]

        subdiag = np.tril_indices(n_nodes, k=-1)
        eta = self.intercept_ + ((self.U_ * self.lambda_) @ self.U_.T)[subdiag]

        if test_indices is not None:
            eta = eta[test_indices]
            y_true = y[test_indices]
        else:
            y_true = y
        
        mu = LINK_FUNCS[self.link](eta)
        dis = get_distribution(mu, dispersion=self.dispersion_, family=self.family)
        loglik = dis.log_prob(y_true).sum()

        return np.asarray(loglik).item()
    
    def predict(self, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : predict(
                glnem, rng_key, samples,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0))
    
    def predict_zero_probas(self, random_state=42):
        rng_key = random.PRNGKey(random_state)
        n_samples  = self.samples_['U'].shape[0]
        vmap_args = (self.samples_, random.split(rng_key, n_samples))

        return np.asarray(vmap(
            lambda samples, rng_key : predict_zero_probas(
                glnem, rng_key, samples,
                *self.model_args_, **self.model_kwargs_)
        )(*vmap_args).mean(axis=0))

    def print_summary(self, proba=0.9):
        print_summary(self,
                self.samples_, self.feature_names_, self.diverging_, prob=proba)

    def plot(self, Y_obs=None, **fig_kwargs):
        return plot_glnem(self, Y_obs, **fig_kwargs)
