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
from glnem.datasets import synthetic_network
from dyneigenmodel.model_selection import train_test_split, kfold


set_host_device_count(10)
numpyro.enable_x64()

family = 'tweedie'
link = 'log'
dispersion = 2

def ic_selection(Y, n_features):

    # fit model
    model = GLNEM(
            family=family, link=link,
            infer_dimension=False, 
            n_features=n_features)

    model.sample(Y, n_warmup=2500, n_samples=2500)

    return n_features, model.waic(), model.dic(), model.aic(Y), model.bic(Y)



Y, X, params = synthetic_network(n_nodes=75, family=family, link=link,
        intercept=0, n_features=3, n_covariates=4, random_state=1,
        dispersion=dispersion, var_power=1.7)

## information criteria
res = Parallel(n_jobs=-1)(delayed(ic_selection)(Y, d) for d in range(1, 9))
data = pd.DataFrame(
    np.asarray(res), columns=['n_features', 'waic', 'dic', 'aic', 'bic'])

## cross-validation
#res = Parallel(n_jobs=-1)(delayed(kfold_selection)(Y, d) for d in range(1, 11))
#data = pd.DataFrame(np.asarray(res), columns=['n_features', 'loglik', 'auc'])

 Spike-and-Slab Variable Selection
model = GLNEM(family=family, link=link, n_features=8, random_state=123)

model.sample(Y, n_warmup=2500, n_samples=2500)

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
