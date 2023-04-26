import numpy as np
import pandas as pd
import networkx as nx

from os.path import dirname, join


__all__ = ['load_ppnetwork']


def load_ppnetwork(connected_component=False):
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data')
    Y = np.loadtxt(join(file_path, 'ecoli.dat'))
    #names = pd.read_csv(join(file_path, 'ecoli_names.dat'), header=None)

    if connected_component:
        G = nx.from_numpy_array(Y)
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        Y = nx.to_numpy_array(G.subgraph(Gcc[0]))
        
    return Y#, names[0].values

