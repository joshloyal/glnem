import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

data_dir_d6 = 'output_ss_d6'
data_dir_d8 = 'output_ss_d8'
data_dir_d12 = 'output_ss_d12'

criteria = ['aic', 'bic', 'dic', 'waic']
family = ['bernoulli', 'gaussian', 'poisson', 'negbinom', 'tweedie']
family_map = {'bernoulli': 'Bernoulli', 'gaussian': 'Gaussian', 'poisson': 'Poisson', 'negbinom': 'Negative Binomial', 'tweedie': 'Tweedie'}
all_methods = ['d6', 'd8', 'd12']
method_names = ['d = 6', 'd = 8', 'd = 12']
#method_names = ['Spike-Slab (Proposed)', 'CV', 'CV (1SE)', 'BIC', 'DIC', 'WAIC']

#fig, ax = plt.subplots(nrows=len(family), ncols=3, sharex=True, figsize=(20, 15))
for j, fam_name in enumerate(family):
    fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(20, 4))
    if fam_name in ['poisson', 'bernoulli', 'gaussian']:
        n_nodes = [100, 200, 300]
    elif fam_name == 'tweedie':
        n_nodes = [50, 100, 150]
    else:
        n_nodes = [100, 150, 200]

    for i, n in enumerate(n_nodes):
        
        data_d6 = []
        for file_name in glob.glob(f"{data_dir_d6}/{fam_name}_n{n}/result_*ss_*.csv"):
            df = pd.read_csv(file_name)
            data_d6.append(['d6', df['d_hat'].values[0]]) 
        
        data_d8 = []
        for file_name in glob.glob(f"{data_dir_d8}/{fam_name}_n{n}/result_*ss_*.csv"):
            df = pd.read_csv(file_name)
            data_d8.append(['d8', df['d_hat'].values[0]]) 
        
        data_d12 = []
        for file_name in glob.glob(f"{data_dir_d12}/{fam_name}_n{n}/result_*ss_*.csv"):
            df = pd.read_csv(file_name)
            data_d12.append(['d12', df['d_hat'].values[0]]) 

        data = []
        
        if len(data_d6):
            data.append(pd.DataFrame(data_d6, columns=['variable', 'value']))
            
        if len(data_d8):
            data.append(pd.DataFrame(data_d8, columns=['variable', 'value']))
        
        if len(data_d12):
            data.append(pd.DataFrame(data_d12, columns=['variable', 'value']))
             
        if len(data):
            data = pd.concat(data) 
            
            unique_values = np.unique(data['value'])
            unique_var = np.unique(data['variable'])

            data = data.groupby(['variable', 'value']).size().reset_index(name='count')
            data = data.pivot_table(index="variable", columns="value", values="count", fill_value=0)
            
            x = np.zeros((len(all_methods), 12))
            for idx, m in enumerate(all_methods):
                jdx = np.argwhere(m == unique_var)
                if len(jdx):
                    x[idx, unique_values-1] = data.values[jdx.item()]
            with np.errstate(divide='ignore'):
                x /= x.sum(axis=1).reshape(-1, 1)
            x[x == 0] = np.nan
            x = pd.DataFrame(100 * x, index=method_names)
            x.columns = np.arange(1, 13) 
            
            g = sns.heatmap(x, annot=True, ax=ax[i], cmap='Greys', vmin=0, vmax=100, fmt='.3g',
                    yticklabels=(i==0), xticklabels=True, cbar=False, annot_kws={'size': 16})
            ax[i].set_yticklabels(ax[i].get_ymajorticklabels(), fontsize=16) 
            ax[i].set_xticklabels(ax[i].get_xmajorticklabels(), fontsize=16) 
            #legend = (i == 2)
            #hue_order = ['ss', 'cv', 'cv_1se'] + criteria
            #sns.histplot(data, x='value', hue='variable',  discrete=True, multiple='dodge', common_norm=False,
            #        binrange=(1, 8), ax=ax[i], shrink=0.8, stat='percent',
            #        legend=legend, hue_order=hue_order)

            #if legend:
            #    sns.move_legend(ax[i], "upper left",
            #            bbox_to_anchor=(1, 1), title=None, frameon=False)
            
            if i == 1:
                ax[i].set_title(f"{family_map[fam_name]}\n\nn = {n}\n", fontsize=20)
            else:
                ax[i].set_title(f"n = {n}\n", fontsize=20)

            #if i == 0:
            #    #ax[i].set_ylabel(f"{family_map[fam_name]}\n\nMethod")
            #    ax[i].set_ylabel("Method")

            #if (j == len(family)-1):
            #    ax[j, i].set_xlabel("Latent Space Dimension")
            ax[i].set_xlabel("Selected dimension", fontsize=16)
            
            ax[i].add_patch(Rectangle((2, 0), 1, len(all_methods)-0.01, fill=False, edgecolor='black', lw=3))

    fig.savefig(f'heatmap_{fam_name}_dim_comp.png', dpi=600, bbox_inches='tight')
