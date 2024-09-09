import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'output'
family = ['gaussian', 'laplace', 't_10.0', 't_5.0', 't_3.0']
str_fmt = r"{} & {:.0f}\% & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) \\"  
table = ''
family_map = {
        'gaussian': 'Gaussian',
        'laplace': 'Laplace',
        't_3.0': r'$t_3$',
        't_5.0': r'$t_5$',
        't_10.0': r'$t_{10}$'
}

for i, fam_name in enumerate(family):
    data = []
    for file_name in glob.glob(f"{data_dir}/{fam_name}_n200/result*.csv"):
        df = pd.read_csv(file_name)
        data.append(df)
    
    name = family_map[fam_name] 
    if len(data):
        data = pd.concat(data)
        percent = (data['d_hat'].values == 3).mean() * 100
        med = data.median(axis=0)#[5:]
        std = data.std(axis=0)#[5:]
        res = pd.concat((med, std), axis=1)
        t = str_fmt.format(name, percent,
                res.loc['d_hat', 0], res.loc['d_hat', 1],
                res.loc['U_corr', 0], res.loc['U_corr', 1],
                res.loc['lambda_rel', 0] * 100 , res.loc['lambda_rel', 1] * 100,
                res.loc['sim_rel', 0] * 100, res.loc['sim_rel', 1] * 100,
                res.loc['coef_rel', 0] * 100, res.loc['coef_rel', 1] * 100)

        table = table + '\n' + t

print(table)
