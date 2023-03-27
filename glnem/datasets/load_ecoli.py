import numpy as np
import pandas as pd

from os.path import dirname, join


__all__ = ['load_ecoli']


def load_ecoli():
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')
    Y = np.loadtxt(join(file_path, 'ecoli.dat'))
    names = pd.read_csv(join(file_path, 'ecoli_names.dat'), header=None)

    return Y, names[0].values

