import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import find, csr_matrix

def get_adj_DM(adata,
        k=15,
        n_jobs=-2,
        pca=None):
    """
    :param data_df: Normalized data frame. 
    :param k: Number of nearest neighbors for graph construction
    :param n_jobs: Nearest Neighbors will be computed in parallel using n_jobs.
    :param pc_components: Minimum number of principal components to use. Specify `None` to use pre-computed components
    :return: Affinity matrix  
    """

    # Nearest neighbor graph construction and affinity matrix
    print('Nearest neighbor computation...')
    countp = adata

    dist = kneighbors_graph(countp, countp.shape[0], mode="distance", metric="euclidean", include_self=True)
    dist_array = dist.toarray()

    # Adaptive k
    adaptive_k = int(np.floor(k))
    scaling_factors = np.zeros(countp.shape[0])

    for i in np.arange(len(scaling_factors)):
        # print(np.sort(dist_array[i,:]))
        scaling_factors[i] = np.sort(dist_array[i,:])[adaptive_k]

    scaling_factors = pd.Series(scaling_factors)

    # Affinity matrix
    nn_aff = _convert_to_affinity(dist, scaling_factors, True)

    # Symmetrize the affinity matrix and return
    # aff = nn_aff + nn_aff.T
    aff = nn_aff
    adj = aff.toarray()

    indexs = adj.argsort() 
    rep_num = adj.shape[1]+1-k
    for i in range(indexs.shape[0]):
        adj[i, indexs[i, :rep_num]]=0
        adj[i, i]=1

    adj[adj>0] = 1
    # print("raw: " + str(np.sum(adj, axis=1)))
    # print("col: " + str(np.sum(adj, axis=0)))
    adj_n = norm_adj(adj)

    return adj, adj_n

def get_adj(adata, k=15, pca=None):
    countp = adata
    A = kneighbors_graph(countp, k, mode="connectivity", metric="euclidean", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    return adj, adj_n


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def _convert_to_affinity(adj, scaling_factors, device, with_self_loops=False):
    """ Convert adjacency matrix to affinity matrix
    """
    N = adj.shape[0]
    rows, cols, dists = find(adj)

    dists = dists ** 2 / (2 * scaling_factors.values[rows] ** 2) +\
            dists ** 2 / (2 * scaling_factors.values[cols] ** 2)

    aff = csr_matrix((np.exp(-dists), (rows, cols)), shape=[N, N])
    return aff
