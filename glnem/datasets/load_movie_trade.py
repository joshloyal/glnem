import numpy as np
import pandas as pd
import networkx as nx
import joblib

from os.path import dirname, join
from glnem.network_utils import adjacency_to_vec


__all__ = ['load_trade']
    

def weights_to_covariates(g, weights, ids=None):
    if ids is None:
        n_nodes = g.number_of_nodes()
    else:
        n_nodes = len(ids)
    
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    X = np.zeros((n_dyads, len(weights)))
    for p, weight in enumerate(weights):
        x = nx.to_numpy_array(g, weight=weight)
        if ids is not None:
            x = x[ids][:, ids]
        
        X[:, p] = adjacency_to_vec(x)

    return X


def load_data(product):
    module_path = dirname(__file__)
    file_path = join(module_path, 'raw_data', 'BACI_HS17_V202301')

    #data = pd.read_csv(join(file_path, f"BACI_HS17_Y{year}_V202301.csv")
    #data = data.query(f"year == {year}")
    
    data = pd.read_csv(join(file_path, f"{product}_gravity.csv"))
    data = data.dropna(subset=['gdp_o', 'gdp_d', 'dist'])
    
    data['gdp_dyad'] = np.log1p(data['gdp_o']) + np.log1p(data['gdp_d'])
    data['dist'] = np.log1p(data['dist'])

    return data


#def load_baci_product(product='movies', n_nodes=50): 
#    data = load_data(product=product)
#
#    g = nx.from_pandas_edgelist(data, source='iso3_o', target='iso3_d',#source='i', target='j', 
#            edge_attr=['v', 'dist', 'gdp_dyad'])
#
#    Y = nx.to_numpy_array(g, weight='v')
#    Y += Y.T
#    
#    # in billions of dollars
#    scale = 100000
#    ids = np.argsort(Y.sum(axis=0))[::-1][:n_nodes]
#    if n_nodes is not None:
#        Y = Y[ids][:, ids] / scale 
#    
#    X = weights_to_covariates(g, ['dist', 'gdp_dyad'], ids=ids)
#
#    return Y, np.log1p(X)


def load_trade(trade_type='movies', max_nodes=None):
    module_path = dirname(__file__)
    base_path = join(module_path, 'raw_data', 'BACI_HS17_V202301')
    A = joblib.load(join(base_path, f'{trade_type}.npy'))
    countries = pd.read_csv(
             join(base_path, f'countries_{trade_type}.csv')).values
    Y = A[0]

    if max_nodes is not None:
        node_ids = np.argsort(Y.sum(axis=1))[::-1][:max_nodes]
        Y = Y[node_ids][:, node_ids]
        countries = countries[node_ids]

    n_nodes = Y.shape[0]
    n_dyads = int(0.5 * n_nodes * (n_nodes - 1))
    n_features = A.shape[0] - 1
    X = np.zeros((n_dyads, n_features))
    for p in range(n_features):
        Ap = A[p+1]
        if max_nodes is not None:
            Ap = Ap[node_ids][:, node_ids]
        X[:, p] = adjacency_to_vec(Ap)

    return Y, X, countries
