import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas as pd
import scipy.stats as stats

from scipy.special import expit
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score
from numpyro.util import set_host_device_count
from joblib import Parallel, delayed

from glnem import GLNEM
from glnem.glnem import find_permutation
from glnem.datasets import synthetic_network
from glnem.model_selection import kfold


set_host_device_count(10)
numpyro.enable_x64()

family = 'negbinom'
link = 'log'
dispersion = 0.5


def compare_latent_space(U_pred, U_true):
    n_nodes, n_features = U_true.shape

    U_pred, perm = find_permutation(U_pred, U_true)
    corr = np.sum(U_pred * U_true) / n_features
    mse = np.sum((U_pred - U_true) ** 2) / np.sum(U_true ** 2)

    return corr, mse, perm


def ic_selection(Y, n_features):

    # fit model
    model = GLNEM(
            family=family, link=link,
            infer_dimension=False, 
            n_features=n_features)

    model.sample(Y, n_warmup=2500, n_samples=2500)

    return n_features, model.waic(), model.dic(), model.aic(Y), model.bic(Y)



Y, X, params = synthetic_network(n_nodes=100, family=family, link=link,
        intercept=-1.0, n_features=3, n_covariates=None, random_state=0,
        dispersion=dispersion, var_power=1.6)
print(Y.mean())

## information criteria
#res = Parallel(n_jobs=-1)(delayed(ic_selection)(Y, d) for d in range(1, 9))
#data = pd.DataFrame(
#    np.asarray(res), columns=['n_features', 'waic', 'dic', 'aic', 'bic'])

## cross-validation
#res = Parallel(n_jobs=-1)(delayed(kfold_selection)(Y, d) for d in range(1, 11))
#data = pd.DataFrame(np.asarray(res), columns=['n_features', 'loglik', 'auc'])

# Spike-and-Slab Variable Selection
model = GLNEM(family=family, link=link, n_features=8, random_state=123)

model.sample(Y, X=X, n_warmup=2500, n_samples=2500)

# dimension selection
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
data['d_hat'] = d_hat[0]

# latent space error
n_features = params['U'].shape[1]
corr, mse, perm = compare_latent_space(model.U_[..., :n_features], params['U'])
data['U_corr'] = corr
data['U_rel'] = mse

# lambda error
lmbda_pred = model.lambda_[:n_features]
lmbda = params['lambda']
data['lambda_rel'] = np.sum((lmbda_pred[perm] - lmbda) ** 2) / np.sum(lmbda ** 2)

# g(E[Y]) error
eta = params['linear_predictor']
eta_pred = model.linear_predictor()
data['linear_predictor_rel'] = np.sum((eta_pred - eta) ** 2) / np.sum(eta ** 2)

# intercept
data['intercept_rel'] = np.sqrt((model.intercept_ - params['intercept']) ** 2) 

# regression coefficients
for p in range(X.shape[1]):
    data[f'coef{p}_rel'] = np.sqrt((model.coefs_[p] - params['coefs'][p]) ** 2) 
