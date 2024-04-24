import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import mode
from statsmodels.graphics.gofplots import qqplot

from .diagnostics import quantile_residuals
from .network_utils import adjacency_to_vec, dynamic_adjacency_to_vec
from .gof import density, std_degree, transitivity
from .gof import degree, degree_distribution


BOXPLOT_PROPS = {
    'boxprops': {'facecolor': 'none', 'edgecolor': 'black'},
    'medianprops': {'color': 'black', 'linewidth': 2},
    'whiskerprops': {'color': 'black', 'linestyle': '--'},
    'capprops': {'color': 'black'}
}


def plot_glnem(glnem, Y_obs=None, include_diagnostics=True, **fig_kwargs):
    if glnem.infer_dimension:
        if include_diagnostics:
            if glnem.X_dyad_ is not None:
                ax = plt.figure(
                    constrained_layout=True, **fig_kwargs).subplot_mosaic(
                    """
                    ABC
                    DEF
                    GHI
                    """
                )
            else:
                ax = plt.figure(
                    constrained_layout=True, **fig_kwargs).subplot_mosaic(
                    """
                    AAB
                    DEF
                    GHI
                    """
                )
        else:
            if glnem.X_dyad_ is not None:
                ax = plt.figure(
                    constrained_layout=True, **fig_kwargs).subplot_mosaic(
                    """
                    ABC
                    DEF
                    """
                )
            else:
                ax = plt.figure(
                    constrained_layout=True, **fig_kwargs).subplot_mosaic(
                    """
                    AAB
                    DEF
                    """
                )
    else:
        if include_diagnostics:
            ax = plt.figure(
                constrained_layout=True, **fig_kwargs).subplot_mosaic(
                """
                ABC
                GHI
                """
            )
        else:
            ax = plt.figure(
                constrained_layout=True, **fig_kwargs).subplot_mosaic(
                """
                ABC
                """
            )

    # plot lambda values
    n_samples, n_features = glnem.samples_['lambda'].shape
    ax['A'].plot(np.asarray(glnem.samples_['lambda']), alpha=0.8)
    for h in range(n_features):
        ax['A'].axhline(glnem.lambda_[h], color='k', linestyle='--', lw=2)
        ax['A'].text(x=n_samples, y=glnem.lambda_[h],
            s=r"$\lambda_{{{}}}$".format(h+1))
    ax['A'].set_ylabel(r'$\Lambda$')
    ax['A'].set_xlabel(r'Iteration')

    # plot intercept
    ax['B'].plot(np.asarray(glnem.samples_['intercept']), alpha=0.8)
    ax['B'].axhline(glnem.intercept_, color='k', linestyle='--', lw=2)
    ax['B'].set_ylabel(r'Intercept')
    ax['B'].set_xlabel(r'Iteration')
    
    if glnem.X_dyad_ is not None: 
        for p, name in enumerate(glnem.feature_names_):
            coef_samples = glnem.samples_[name] 
            ax['C'].plot(coef_samples, alpha=0.8)
            ax['C'].axhline(coef_samples.mean(), color='k', linestyle='--')
            ax['C'].text(x=n_samples, y=coef_samples.mean(), s=r"$\beta_{}$".format(p+1)) 
        ax['C'].set_ylabel(r'Covariate Effects')
        ax['C'].set_xlabel(r'Iteration')

    # plot dimension selection
    if glnem.infer_dimension:
        # trace plot of dimension
        ax['D'].plot(np.asarray(glnem.samples_['s'].sum(axis=1)), alpha=0.8)
        ax['D'].axhline(mode(glnem.samples_['s'].sum(axis=1), keepdims=False).mode,
            color='k', linestyle='--', lw=2)
        ax['D'].set_xlabel(r'Iteration')
        ax['D'].set_ylabel(r'# of Dimensions')

        dimensions = np.arange(1, glnem.n_features + 1)

        # posterior # of dimensions
        K_eff = np.bincount(
            glnem.samples_['s'].sum(axis=1).astype(int),
            minlength=glnem.n_features+1)[1:]
        ax['E'].bar(
            dimensions, height=K_eff / K_eff.sum(), width=0.25, color='#77a8cd', edgecolor='k')
        ax['E'].set_xlim(1, glnem.n_features + 1)
        ax['E'].set_xticks(dimensions)
        ax['E'].set_xlabel('# of Dimension')
        ax['E'].set_ylabel('Posterior Probability')

        # posterior inclusion probabilities
        ax['F'].plot(dimensions,
            glnem.inclusion_probas_, 'ko')
        ax['F'].axhline(0.5, color='k', linestyle='--', lw=2)
        ax['F'].set_xticks(dimensions)
        ax['F'].set_xlabel('Latent Dimension')
        ax['F'].set_ylabel('Posterior Inclusion Probability')

    # goodness-of-fit statistics
    y_vec = adjacency_to_vec(Y_obs)
    
    if not include_diagnostics:
        return ax

    if glnem.family == 'bernoulli':
        stats = {
            'density': density,
            'transitivity': transitivity
        }
        names = ['G', 'H']
        for k, (key, stat_func) in zip(names, stats.items()):
            res = glnem.posterior_predictive(stat_func)
            sns.histplot(res, edgecolor='k', color='#add8e6', ax=ax[k])
            ax[k].axvline(
                stat_func(y_vec), color='k', linestyle='--', linewidth=3)
            ax[k].set_xlabel(key)

        # degree distribution
        degrees = glnem.posterior_predictive(degree)
        max_degree = np.max(degrees) + 1
        deg_dist = degree_distribution(degrees)
        sns.boxplot(x='degree', y='count', data=deg_dist, color='w', fliersize=0,
                    ax=ax['I'], **BOXPLOT_PROPS)
        ax['I'].plot(
            np.bincount(degree(y_vec).ravel(), minlength=max_degree + 1),
            'k-', linewidth=3)

        # 95% credible intervals
        bounds = deg_dist.groupby('degree').quantile([0.025, 0.975])
        ax['I'].plot(bounds.xs(0.025, level=1).values.ravel(), ':', c='gray')
        ax['I'].plot(bounds.xs(0.975, level=1).values.ravel(), ':', c='gray')
        ax['I'].set_ylabel('Number of Nodes')
        ax['I'].set_xlabel('Degree')
    else:
        # diagnostic plots 
        y_hat = glnem.predict()
        
        if glnem.family in ['poisson', 'negbinom', 'tweedie', 'tobit']:
            # fitted vs. quantile residual plot for distributions with discrete components
            dispersion = glnem.samples_['dispersion'].mean() if glnem.family in ['negbinom', 'tweedie', 'tobit'] else 0
            var_power = glnem.samples_['var_power'].mean() if glnem.family == 'tweedie' else None
            resid = quantile_residuals(y_vec, y_hat, dispersion=dispersion, var_power=var_power, family=glnem.family)
            ylab = 'Quantile Residual'
        else:
            # "standardized residuals" using the estimated dispersion
            if glnem.dispersion_ is not None:
                resid = (y_vec - y_hat) / glnem.dispersion_
            else:
                resid = y_vec - y_hat 
            ylab = 'Standardized Residual'
        
        # fitted vs. residual
        ax['G'].scatter(y_hat, resid, s=5)
        ax['G'].axhline(0, linestyle='--', c='k')
        ax['G'].set_ylabel(ylab)
        ax['G'].set_xlabel('Fitted')
        
        qqplot(resid, line='45', markersize=2, ax=ax['H'])

        # qq plot
        if glnem.family in ['poisson', 'negbinom', 'tweedie', 'tobit']:
            # AUC curve for distinguishing zeros
            probas = glnem.predict_zero_probas()
            y_bin = y_vec == 0

            fpr, tpr, _ = roc_curve(y_bin, probas)
            try:
                auc = roc_auc_score(y_bin, probas)
                ax['I'].plot(fpr, tpr)
                ax['I'].plot([0, 1], [0, 1], 'k--')
                ax['I'].annotate(f'AUC = {auc:.3f}', (0.25, 0.0))
                ax['I'].set_ylabel('TPR')
                ax['I'].set_xlabel('FPR')
            except:
                pass

    return ax


def plot_covariate_posteriors(glnem, var_names=None, var_labels=None, figsize=None, ax=None):
    if var_names is None:
        var_names = glnem.feature_names_
    else:
        var_names = [v for v in var_names if v in glnem.feature_names_]

    if var_labels is None:
        var_labels = var_names
    
    if len(var_names) == 0 or len(var_names) != len(var_labels):
        raise ValueError
    
    if ax is None:
        figsize = (4 * len(var_names), 2) if figsize is None else figsize 
        fig, ax = plt.subplots(ncols=len(var_names), figsize=figsize)

    for k, var in enumerate(var_names):
        interval = list(np.quantile(glnem.samples_[var], q=[0.025, 0.975]))
        mean = glnem.samples_[var].mean()

        sns.kdeplot(glnem.samples_[var], c='k', ax=ax[k])
        ax[k].plot([interval[0], interval[1]], [0, 0], lw=5, c='lightgray')

        if k == 1:
            ax[k].text(0.5, 0.8, f"[{interval[0]:.2f}, {interval[1]:.2f}]",  
                    transform=ax[k].transAxes)
        else:
            ax[k].text(0.65, 0.8, f"[{interval[0]:.2f}, {interval[1]:.2f}]",  
                    transform=ax[k].transAxes)

        ax[k].axvline(0, c='k', linestyle='--', alpha=0.3)
        ax[k].set_xlabel(var_labels[k], fontsize=12)
        ax[k].set_ylabel('Density', fontsize=12)
        if k > 0:
            ax[k].set_ylabel('')

    return ax
