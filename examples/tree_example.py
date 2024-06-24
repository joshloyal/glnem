import matplotlib.pyplot as plt

from glnem import GLNEM
from glnem.datasets import load_trees


Y, X = load_trees()

# weighted adjacency matrix
Y.shape
#>>> (51, 51)

# dyadic covariates, shape (n_dyads, n_covariates)
X.shape
#>>> (1275, 3)

# initialize negative binomial GLNEM with a log-link and d = 10 latent features
glnem = GLNEM(family='negbinom', link='log', n_features=10)

# run the MCMC algorithm for 2,500 warm-up iterations and collect 2,500 post warm-up samples
glnem.sample(Y, X, n_warmup=2500, n_samples=2500)

# summary of the posterior distribution
glnem.print_summary()

# diagnostic plots
glnem.plot(Y_obs=Y, figsize=(12, 9))
plt.savefig("nb_diag.png", dpi=300, bbox_inches='tight')
