import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plac

n_nodes = [100, 200]

family_map = {
        'bernoulli': 'Bernoulli',
        'gaussian': 'Gaussian',
        'poisson': 'Poisson',
}

model_map = {
        'glnem': 'GLNEM',
        'lpm': 'Distance',
        'lsm': 'Bilinear'
}

def make_table(ls_type):
    data_dir = f'output_{ls_type}'
    str_fmt = r"{} & {} & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) \\"  
    table = ''
    for family in ['bernoulli', 'gaussian']:
        for i, n in enumerate(n_nodes):
            data = []
            for file_name in glob.glob(f"{data_dir}/{family}_n{n}/result_*.csv"):
                data.append(pd.read_csv(file_name))
            
            name = family_map[family] if i == 0 else ''
            if len(data):
                data = pd.concat(data)
                table = table + '\n' + name + '\n'
                data['x_coef_rel'] = data.apply(lambda x: x['coef0_rel'] ** 2 + x['coef1_rel'] ** 2 + x['coef2_rel'] ** 2 + x['coef3_rel'] ** 2 / (0.5 ** 2 + 0.5 ** 2), axis=1)
                if family == 'bernoulli':
                    for model in ['glnem', 'lsm', 'lpm']:
                        med = data.median(axis=0)[5:]
                        std = data.std(axis=0)[5:]
                        res = pd.concat((med, std), axis=1)
                        nn = n if model == 'lsm' else ''
                        t = str_fmt.format(nn, model_map[model], 
                                res.loc[f'{model}_ppc', 0], res.loc[f'{model}_ppc', 1],
                                res.loc[f'{model}_train_auc', 0], res.loc[f'{model}_train_auc', 1],
                                res.loc[f'{model}_test_auc', 0], res.loc[f'{model}_test_auc', 1])

                        table = table + '\n' + t
                elif family == 'poisson':
                    for model in ['glnem', 'lsm', 'lpm']:
                        med = data.median(axis=0)[5:]
                        std = data.std(axis=0)[5:]
                        res = pd.concat((med, std), axis=1)
                        nn = n if model == 'lsm' else ''
                        t = str_fmt.format(nn, model_map[model], 
                                res.loc[f'{model}_ppc', 0], res.loc[f'{model}_ppc', 1],
                                res.loc[f'{model}_train_mse', 0], res.loc[f'{model}_train_mse', 1],
                                res.loc[f'{model}_test_mse', 0], res.loc[f'{model}_test_mse', 1])
                    
                        table = table + '\n' + t
                else:
                    for model in ['glnem', 'lsm', 'lpm']:
                        med = data.median(axis=0)[5:]
                        std = data.std(axis=0)[5:]
                        res = pd.concat((med, std), axis=1)
                        nn = n if model == 'lsm' else ''
                        t = str_fmt.format(nn, model_map[model], 
                                res.loc[f'{model}_ppc', 0], res.loc[f'{model}_ppc', 1],
                                res.loc[f'{model}_train_r2', 0], res.loc[f'{model}_train_r2', 1],
                                res.loc[f'{model}_test_r2', 0], res.loc[f'{model}_test_r2', 1])

                        table = table + '\n' + t
    print(table)

if __name__ == '__main__':
    plac.call(make_table)
