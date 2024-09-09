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


def simulation(seed, n_nodes=100, zif_prob=0.1, family='negbinom'):
    seed = int(seed)
    n_nodes = int(n_nodes)
    zif_prob = float(zif_prob)

    intercept = -1
    dispersion = 0.5
    link = 'log'

    Y, X, params = synthetic_network(n_nodes=n_nodes, family='zif_poisson', link=link,
            intercept=intercept, n_features=3, n_covariates=4, random_state=seed,
            dispersion=dispersion, var_power=1.6, zif_prob=zif_prob)

    model = GLNEM(family=family, link=link, n_features=8, random_state=123)

    model.sample(Y, X=X, n_warmup=n_warmup, n_samples=n_samples)

    s = model.samples_['s'].sum(axis=1)
    d_mean = s.mean()
    d_hat = stats.mode(s)[0]

    d_post = np.bincount(
        model.samples_['s'].sum(axis=1).astype(int),
        minlength=model.n_features+1)[1:]
    d_post = d_post / model.samples_['s'].shape[0]

    data = pd.DataFrame({
        'd': np.arange(1, 9),
        'inclusion_probas': model.inclusion_probas_,
        'd_post': d_post})
    data['d_mean'] = d_mean
    data['d_hat'] = d_hat

    # latent space error
    n_features = params['U'].shape[1]
    corr, mse, perm = compare_latent_space(model.U_[..., :n_features], params['U'])
    data['U_corr'] = corr

    # lambda error
    lmbda_pred = model.lambda_[:n_features]
    lmbda = params['lambda']
    data['lambda_rel'] = np.sum((lmbda_pred[perm] - lmbda) ** 2) / np.sum(lmbda ** 2)
    # intercept
    data['intercept_rel'] = np.sqrt((model.intercept_ - params['intercept']) ** 2)

    # regression coefficients
    for p in range(X.shape[1]):
        data[f'coef{p}_rel'] = np.sqrt((model.coefs_[p] - params['coefs'][p]) ** 2)

    # coefs relative error
    num = np.sum((model.coefs_ - params['coefs']) ** 2) + (model.intercept_ - params['intercept']) ** 2
    dem = np.sum(params['coefs'] ** 2) + params['intercept'] ** 2
    data['coef_rel'] = num / dem

    # ULUt relative error
    sims = model.similarities()
    data['sim_rel'] = np.sum((sims - params['similarities']) ** 2) / np.sum(params['similarities'] ** 2)

    out_file = f'result_{family}_ss_{seed}.csv'
    dir_base = f'output_zero_inflation/output'
    dir_name = os.path.join(dir_base, f"{family}_n{n_nodes}_p{zif_prob}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 1

for n_nodes in [25, 200]:
    for family in ['poisson', 'negbinom']:
        for i in range(n_reps):
            simulation(seed=i, n_nodes=n_nodes, family=family, zif_prob=0.1)
