import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plac


def make_table(n_features=8):
    data_dir = f'output_ss_d{n_features}'

    family_map = {
            'bernoulli': 'Bernoulli',
            'gaussian': 'Gaussian',
            'poisson': 'Poisson',
            'negbinom': 'Neg. Bin.',
            'tweedie': 'Tweedie'
    }

    str_fmt = r"{} & {} & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) \\"  
    table = ''
    for family in ['bernoulli', 'gaussian', 'poisson', 'negbinom', 'tweedie']:
        if family in ['bernoulli', 'gaussian', 'poisson']:
            n_nodes = [100, 200, 300]
        elif family == 'negbinom':
            n_nodes = [100, 150, 200]
        else:
            n_nodes = [50, 100, 150]

        for i, n in enumerate(n_nodes):
            data = []
            for file_name in glob.glob(f"{data_dir}/{family}_n{n}/result_*ss_*.csv"):
                data.append(pd.read_csv(file_name))
            
            name = family_map[family] if i == 1 else ''
            if len(data):
                data = pd.concat(data)
                med = data.median(axis=0)[5:]
                std = data.std(axis=0)[5:]
                res = pd.concat((med, std), axis=1)
                t = str_fmt.format(name, n, 
                        res.loc['U_corr', 0], res.loc['U_corr', 1],
                        res.loc['lambda_rel', 0] * 100 , res.loc['lambda_rel', 1] * 100,
                        res.loc['sim_rel', 0] * 100, res.loc['sim_rel', 1] * 100,
                        res.loc['coef_rel', 0] * 100, res.loc['coef_rel', 1] * 100)
                if i == 2:
                    t = t + "[0.5em]\n"

                table = table + '\n' + t

    print(table)


if __name__ == '__main__':
    plac.call(make_table)
