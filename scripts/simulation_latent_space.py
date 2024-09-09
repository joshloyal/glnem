import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import scipy.stats as stats
import plac
import os

from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_poisson_deviance
from numpyro.util import set_host_device_count
from joblib import Parallel, delayed

from glnem import GLNEM, LatentPositionModel
from glnem.glnem import find_permutation
from glnem.datasets import synthetic_lpm_network
from glnem.model_selection import train_test_split
from glnem.network_utils import adjacency_to_vec


set_host_device_count(10)
numpyro.enable_x64()


n_warmup = 5000
n_samples = 5000


def poisson_d2(y_true, y_pred):
    num = mean_poisson_deviance(y_true, y_pred)
    dem = mean_poisson_deviance(
        y_true, np.repeat(y_true.mean(), y_true.shape[0]))

    return 1 - num / dem


def simulation(seed, family, ls_type, n_nodes):
    seed = int(seed)
    n_nodes = int(n_nodes)
    ls_type = ls_type
    family = family
    
    if ls_type in ['distance', 'distance_sq']:
        intercept = 0.5 if family == 'bernoulli' else 1.0
    else:
        if family == 'bernoulli':
            intercept = -2.0
        elif family == 'gaussian':
            intercept = 1.0 
        else:
            intercept = -1.0

    dispersion = 0.5

    if family == 'bernoulli':
        link = 'logit'
    elif family == 'gaussian':
        link = 'identity'
    else:
        link = 'log'

    Y, X, params = synthetic_lpm_network(n_nodes=n_nodes, family=family, 
            link=link, intercept=intercept, n_features='mixture', n_covariates=4, 
            dispersion=dispersion, random_state=seed, ls_type=ls_type) 
    
    train, test = train_test_split(Y, random_state=seed)

    model = GLNEM(family=family, link=link, n_features=8, random_state=123)
    model.sample(Y, X=X, missing_edges=test, n_warmup=n_warmup, n_samples=n_samples)

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

    # g(E[Y]) error
    eta = params['linear_predictor']
    eta_pred = model.linear_predictor()
    data['linear_predictor_rel'] = np.sum((eta_pred - eta) ** 2) / np.sum(eta ** 2)

    # intercept
    data['intercept_rel'] = np.sqrt((model.intercept_ - params['intercept']) ** 2)

    # regression coefficients
    for p in range(X.shape[1]):
        data[f'coef{p}_rel'] = np.sqrt((model.coefs_[p] - params['coefs'][p]) ** 2)

    # coefs relative error
    num = np.sum((model.coefs_ - params['coefs']) ** 2) + (model.intercept_ - params['intercept']) ** 2
    dem = np.sum(params['coefs'] ** 2) + params['intercept'] ** 2
    data['coef_rel'] = num / dem
    
    num = np.sum((model.coefs_ - params['coefs']) ** 2) 
    dem = np.sum(params['coefs'] ** 2) 
    data['x_coef_rel'] = num / dem

    # correlation between means
    mu = model.predict()
    data['glnem_ppc'] = np.corrcoef(params['mu'], mu)[0, 1]
    data['glnem_train_ppc'] = np.corrcoef(params['mu'][train], mu[train])[0, 1]
    data['glnem_test_ppc'] = np.corrcoef(params['mu'][test], mu[test])[0, 1]

    # ULUt relative error
    sims = model.similarities()
    data['sim_rel'] = np.sum((sims - params['similarities']) ** 2) / np.sum(params['similarities'] ** 2)
    
    y = adjacency_to_vec(Y)

    # testing AUC
    if family == 'bernoulli':
        data['glnem_train_auc'] = roc_auc_score(y[train], mu[train])
        data['glnem_test_auc'] = roc_auc_score(y[test], mu[test])
    elif family == 'gaussian':
        data['glnem_train_mse'] = mean_squared_error(y[train], mu[train])
        data['glnem_train_r2'] = r2_score(y[train], mu[train])
        data['glnem_test_mse'] = mean_squared_error(y[test], mu[test])
        data['glnem_test_r2'] = r2_score(y[test], mu[test])
    elif family == 'poisson':
        data['glnem_train_mse'] = mean_squared_error(y[train], mu[train])
        data['glnem_train_r2'] = r2_score(y[train], mu[train])
        data['glnem_train_d2'] = poisson_d2(y[train], mu[train])
        data['glnem_test_mse'] = mean_squared_error(y[test], mu[test])
        data['glnem_test_d2'] = poisson_d2(y[test], mu[test])
    
    model_lpm = LatentPositionModel(
        family=family, link=link, n_features=2, random_state=123)
    model_lpm.sample(Y, X=X, missing_edges=test, n_warmup=n_warmup, n_samples=n_samples)
    
    mu = model_lpm.predict()
    data['lpm_ppc'] = np.corrcoef(params['mu'], mu)[0, 1]
    data['lpm_train_ppc'] = np.corrcoef(params['mu'][train], mu[train])[0, 1]
    data['lpm_test_ppc'] = np.corrcoef(params['mu'][test], mu[test])[0, 1]
    
    if family == 'bernoulli':
        data['lpm_train_auc'] = roc_auc_score(y[train], mu[train])
        data['lpm_test_auc'] = roc_auc_score(y[test], mu[test])
    elif family == 'gaussian':
        data['lpm_train_mse'] = mean_squared_error(y[train], mu[train])
        data['lpm_train_r2'] = r2_score(y[train], mu[train])
        data['lpm_test_mse'] = mean_squared_error(y[test], mu[test])
        data['lpm_test_r2'] = r2_score(y[test], mu[test])
    elif family == 'poisson':
        data['lpm_train_mse'] = mean_squared_error(y[train], mu[train])
        data['lpm_train_r2'] = r2_score(y[train], mu[train])
        data['lpm_test_d2'] = poisson_d2(y[test], mu[test])
        data['lpm_test_mse'] = mean_squared_error(y[test], mu[test])
    
    model_lsm = LatentPositionModel(
        ls_type='bilinear', family=family, link=link, n_features=2, random_state=123)
    model_lsm.sample(Y, X=X, missing_edges=test, n_warmup=n_warmup, n_samples=n_samples)
    
    mu = model_lsm.predict()
    data['lsm_ppc'] = np.corrcoef(params['mu'], mu)[0, 1]
    data['lsm_train_ppc'] = np.corrcoef(params['mu'][train], mu[train])[0, 1]
    data['lsm_test_ppc'] = np.corrcoef(params['mu'][test], mu[test])[0, 1]
    
    if family == 'bernoulli':
        data['lsm_train_auc'] = roc_auc_score(y[train], mu[train])
        data['lsm_test_auc'] = roc_auc_score(y[test], mu[test])
    elif family == 'gaussian':
        data['lsm_train_mse'] = mean_squared_error(y[train], mu[train])
        data['lsm_train_r2'] = r2_score(y[train], mu[train])
        data['lsm_test_mse'] = mean_squared_error(y[test], mu[test])
        data['lsm_test_r2'] = r2_score(y[test], mu[test])
    elif family == 'poisson':
        data['lsm_train_mse'] = mean_squared_error(y[train], mu[train])
        data['lsm_train_r2'] = r2_score(y[train], mu[train])
        data['lsm_train_d2'] = poisson_d2(y[train], mu[train])
        data['lsm_test_mse'] = mean_squared_error(y[test], mu[test])
        data['lsm_test_d2'] = poisson_d2(y[test], mu[test])
        data['lsm_test_d2'] = poisson_d2(y[test], mu[test])

    out_file = f'result_{family}_{seed}.csv'
    dir_base = f'output_latent_space/output_{ls_type}'
    dir_name = os.path.join(dir_base, f"{family}_n{n_nodes}")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data.to_csv(os.path.join(dir_name, out_file), index=False)


# NOTE: This is meant to be run in parallel on a computer cluster!
n_reps = 50

for family in ['gaussian', 'bernoulli']:
    for ls_type in ['bilinear', 'distance', 'distance_sq']:
        for n_nodes in [100, 200]:
            for i in range(n_reps): 
                simulation(seed=i, family=family, ls_type=ls_type, n_nodes=n_nodes)
