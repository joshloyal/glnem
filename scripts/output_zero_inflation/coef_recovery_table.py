import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'output'

family_map = {
        'bernoulli': 'Bernoulli',
        'gaussian': 'Gaussian',
        'poisson': 'Poisson',
        'negbinom': 'Neg. Bin.',
        'tweedie': 'Tweedie'
}

str_fmt = r"{} & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) \\"  
table = ''
for family in ['negbinom']:
    for i, n in enumerate([100, 200]):
        data = []
        for file_name in glob.glob(f"{data_dir}/{family}_n{n}_p0.1/result_*ss_*.csv"):
            data.append(pd.read_csv(file_name))
        
        name = family_map[family] if i == 1 else ''
        if len(data):
            data = pd.concat(data)
            med = data.median(axis=0)[5:]
            std = data.std(axis=0)[5:]
            res = pd.concat((med, std), axis=1)  * 100
            t = str_fmt.format(n, 
                    res.loc['intercept_rel', 0], res.loc['intercept_rel', 1],
                    res.loc['coef0_rel', 0], res.loc['coef0_rel', 1],
                    res.loc['coef1_rel', 0], res.loc['coef1_rel', 1],
                    res.loc['coef2_rel', 0], res.loc['coef2_rel', 1],
                    res.loc['coef3_rel', 0], res.loc['coef3_rel', 1])
            if i == 2:
                t = t + "\n\\bottomrule"

            table = table + '\n' + t

print(table)
