import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import scipy.stats as stats
import plac
import os

from scipy.special import expit
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score
from numpyro.util import set_host_device_count
from joblib import Parallel, delayed

from glnem import GLNEM
from glnem.glnem import find_permutation
from glnem.datasets import synthetic_network
from glnem.model_selection import kfold_selection, ic_selection


set_host_device_count(10)
numpyro.enable_x64()


n_warmup = 5000
n_samples = 5000


def compare_latent_space(U_pred, U_true):
    n_nodes, n_features = U_true.shape

    U_pred, perm = find_permutation(U_pred, U_true)
    corr = np.sum(U_pred * U_true) / n_features
    mse = np.sum((U_pred - U_true) ** 2) / np.sum(U_true ** 2)

    return corr, mse, perm


def simulation(seed, n_nodes=100, family='bernoulli', max_features=8):
    seed = int(seed)
    n_nodes = int(n_nodes)
    max_features = int(max_features)

    intercept = 1 if family == 'gaussian' else -1
    if family == 'tweedie':
        dispersion = 10.0
    elif family == 'gaussian':
        dispersion = 3.0
    else:
        dispersion = 0.5

    if family == 'bernoulli':
        link = 'logit'
    elif family == 'gaussian':
        link = 'identity'
    else:
        link = 'log'

    Y, X, params = synthetic_network(n_nodes=n_nodes, family=family, link=link,
            intercept=intercept, n_features=3, n_covariates=4, random_state=seed,
            dispersion=dispersion, var_power=1.6, zif_prob=0.1)

    model_family = 'poisson' if family == 'zif_poisson' else family

    model = GLNEM(family=model_family, link=link, n_features=max_features, random_state=123)

    model.sample(Y, X=X, n_warmup=n_warmup, n_samples=n_samples)
    
    # posterior dimension summaries
    s = model.samples_['s'].sum(axis=1)
    d_mean = s.mean()
    d_hat = stats.mode(s)[0]

    d_post = np.bincount(
        model.samples_['s'].sum(axis=1).astype(int),
        minlength=model.n_features+1)[1:]
    d_post = d_post / model.samples_['s'].shape[0]

    data = pd.DataFrame({
        'd': np.arange(1, max_features + 1),
        'inclusion_probas': model.inclusion_probas_,
        'd_post': d_post})
    data['d_mean'] = d_mean
    data['d_hat'] = d_hat

    # latent space error
    n_features = params['U'].shape[1]
    corr, mse, perm = compare_latent_space(model.U_[..., :n_features], params['U'])
    data['U_corr'] = corr
    data['U_rel'] = mse

    # lambda error
    lmbda_pred = model.lambda_[:n_features]
    lmbda = params['lambda']
    data['lambda_rel'] = np.sum((lmbda_pred[perm] - lmbda) ** 2) / np.sum(lmbda ** 2)

    # coefs relative error
    num = np.sum((model.coefs_ - params['coefs']) ** 2) + (model.intercept_ - params['intercept']) ** 2
    dem = np.sum(params['coefs'] ** 2) + params['intercept'] ** 2
    data['coef_rel'] = num / dem

    # ULUt relative error
    sims = model.similarities()
    data['sim_rel'] = np.sum((sims - params['similarities']) ** 2) / np.sum(params['similarities'] ** 2)

    out_file = f'result_{family}_ss_{seed}.csv'
    dir_base = f'output_ss_d{max_features}'
    dir_name = os.path.join('output_recovery_dim_select_sensitivity', dir_base, f"{family}_n{n_nodes}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

node_map = {
    'bernoulli': [100, 200, 300],
    'gaussian': [100, 200, 300],
    'poisson': [100, 200, 300],
    'negbinom': [100, 150, 200],
    'tweedie': [50, 100, 150]
}

for max_features in [6, 8, 12]:
    # spike and slab prior
    for family in ['bernoulli', 'gaussian', 'poisson', 'negbinom', 'tweedie']:
        for n_nodes in node_map[family]:
            for i in range(n_reps):
                simulation(seed=i, n_nodes=n_nodes, family=family, select_type='ss', max_features=max_features)
