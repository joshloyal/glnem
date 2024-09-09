import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plac

family_map = {
        'bernoulli': 'Bernoulli',
        'gaussian': 'Gaussian',
        'poisson': 'Poisson',
        'negbinom': 'Neg. Bin.',
        'tweedie': 'Tweedie'
}

def make_table(ls_type):
    data_dir = f'output_{ls_type}'
    str_fmt = r"{} & {} & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) \\"  
    table = ''
    for family in ['bernoulli', 'gaussian']:
        n_nodes = [100, 200]

        for i, n in enumerate(n_nodes):
            data = []
            for file_name in glob.glob(f"{data_dir}/{family}_n{n}/result_*.csv"):
                data.append(pd.read_csv(file_name))
            
            name = family_map[family] if i == 0 else ''
            if len(data):
                data = pd.concat(data)
                med = data.median(axis=0)[5:]
                std = data.std(axis=0)[5:]
                res = pd.concat((med, std), axis=1)  * 100
                t = str_fmt.format(name, n, 
                        res.loc['intercept_rel', 0], res.loc['intercept_rel', 1],
                        res.loc['coef0_rel', 0], res.loc['coef0_rel', 1],
                        res.loc['coef1_rel', 0], res.loc['coef1_rel', 1],
                        res.loc['coef2_rel', 0], res.loc['coef2_rel', 1],
                        res.loc['coef3_rel', 0], res.loc['coef3_rel', 1])
                if i == 2:
                    t = t + "\n\\bottomrule"

                table = table + '\n' + t

    print(table)

if __name__ == '__main__':
    plac.call(make_table)
