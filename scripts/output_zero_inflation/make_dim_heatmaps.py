import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


all_methods = ['negbinom', 'poisson']
method_names = ['Neg. Bin.', 'Poisson']

fig, ax = plt.subplots(ncols=2, sharex=True, figsize=(12, 3))
n_nodes = [100, 200]

for i, n in enumerate(n_nodes):
    data_poisson = []
    for file_name in glob.glob(f"output/poisson_n{n}_p0.1/result_*ss_*.csv"):
        df = pd.read_csv(file_name)
        data_poisson.append(['poisson', df['d_hat'].values[0]]) 
    
    data_negbinom = []
    for file_name in glob.glob(f"output/negbinom_n{n}_p0.1/result_*ss_*.csv"):
        df = pd.read_csv(file_name)
        data_negbinom.append(['negbinom', df['d_hat'].values[0]]) 

    data = []
    data.append(pd.DataFrame(data_poisson, columns=['variable', 'value']))
    data.append(pd.DataFrame(data_negbinom, columns=['variable', 'value']))
        
    data = pd.concat(data) 
    
    unique_values = np.unique(data['value'])
    unique_var = np.unique(data['variable'])

    data = data.groupby(['variable', 'value']).size().reset_index(name='count')
    data = data.pivot_table(index="variable", columns="value", values="count", fill_value=0)

    x = np.zeros((len(all_methods), 8))
    for idx, m in enumerate(all_methods):
        jdx = np.argwhere(m == unique_var)
        if len(jdx):
            x[idx, unique_values-1] = data.values[jdx.item()]
    with np.errstate(divide='ignore'):
        x /= x.sum(axis=1).reshape(-1, 1)
    x[x == 0] = np.nan
    x = pd.DataFrame(100 * x, index=method_names)
    x.columns = np.arange(1, 9)   
    
    g = sns.heatmap(x, annot=True, ax=ax[i], cmap='Greys', vmin=0, vmax=100, fmt='.3g',
            yticklabels=(i==0), xticklabels=True, cbar=False, annot_kws={'size': 16})
    ax[i].set_yticklabels(ax[i].get_ymajorticklabels(), fontsize=14) 
    ax[i].set_xticklabels(ax[i].get_xmajorticklabels(), fontsize=14)     
    ax[i].set_title(f"n = {n}\n", fontsize=16)
    ax[i].set_xlabel("Selected dimension", fontsize=14)
    
    ax[i].add_patch(Rectangle((2, 0), 1, len(all_methods)-0.01, fill=False, edgecolor='black', lw=3))

fig.savefig(f'heatmap_zip.png', dpi=300, bbox_inches='tight')
