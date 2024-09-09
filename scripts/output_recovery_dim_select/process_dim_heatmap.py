import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plac

from matplotlib.patches import Rectangle


def make_heatmaps(n_features=8):
    data_dir_ic = f'output_ic_d{n_features}'
    data_dir_ss = f'output_ss_d{n_features}'
    data_dir_cv = f'output_cv_d{n_features}'

    criteria = ['aic', 'bic', 'dic', 'waic']
    family = ['bernoulli', 'gaussian', 'poisson', 'negbinom', 'tweedie']
    family_map = {
            'bernoulli': 'Bernoulli', 
            'gaussian': 'Gaussian', 
            'poisson': 'Poisson', 
            'negbinom': 'Negative Binomial', 
            'tweedie': 'Tweedie'
    }
    all_methods = ['ss', 'cv', 'cv_1se', 'aic', 'bic', 'dic', 'waic']
    method_names = ['SS-IBP (Proposed)', 'K-fold CV', 'K-fold CV 1SE', 'AIC', 'BIC', 'DIC', 'WAIC']

    for j, fam_name in enumerate(family):
        fig, ax = plt.subplots(ncols=3, sharex=True, figsize=(20, 4))
        if fam_name in ['poisson', 'bernoulli', 'gaussian']:
            n_nodes = [100, 200, 300]
        elif fam_name == 'tweedie':
            n_nodes = [50, 100, 150]
        else:
            n_nodes = [100, 150, 200]

        for i, n in enumerate(n_nodes):
            
            data_ic = []
            for file_name in glob.glob(f"{data_dir_ic}/{fam_name}_n{n}/result_*ic_*.csv"):
                data_ic.append(np.argmin(pd.read_csv(file_name).values[..., 1:], axis=0) + 1)

            data_ss = []
            for file_name in glob.glob(f"{data_dir_ss}/{fam_name}_n{n}/result_*ss_*.csv"):
                df = pd.read_csv(file_name)
                data_ss.append(['ss', df['d_hat'].values[0]]) 

            data_cv_1se = []
            data_cv = []
            for file_name in glob.glob(f"{data_dir_cv}/{fam_name}_n{n}/result_*cv_*.csv"):
                x = pd.read_csv(file_name).values[:, 1:]
                best_id = np.argmax(x[:, 0], axis=0)
                data_cv.append(['cv', best_id + 1])
                
                best_1se = np.where(x[:, 0] > x[best_id, 0] - x[best_id, 1])[0][0]
                data_cv_1se.append(['cv_1se', best_1se + 1])
     
            data = []
            if len(data_ic):
                data.append(pd.melt(
                        pd.DataFrame(np.vstack(data_ic), columns=['waic', 'dic', 'aic', 'bic'])[criteria]))
            if len(data_ss):
                data.append(pd.DataFrame(data_ss, columns=['variable', 'value']))
                
            if len(data_cv):
                data.append(pd.DataFrame(data_cv, columns=['variable', 'value']))
                data.append(pd.DataFrame(data_cv_1se, columns=['variable', 'value']))
            
            if len(data):
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
                ax[i].set_yticklabels(ax[i].get_ymajorticklabels(), fontsize=16) 
                ax[i].set_xticklabels(ax[i].get_xmajorticklabels(), fontsize=16) 
                
                if i == 1:
                    ax[i].set_title(f"{family_map[fam_name]}\n\nn = {n}\n", fontsize=20)
                else:
                    ax[i].set_title(f"n = {n}\n", fontsize=20)

                ax[i].set_xlabel("Selected dimension", fontsize=16)
                
                ax[i].add_patch(Rectangle((2, 0), 1, len(all_methods)-0.01, fill=False, edgecolor='black', lw=3))

        fig.savefig(f'heatmap_{fam_name}.png', dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    plac.call(make_heatmaps)
