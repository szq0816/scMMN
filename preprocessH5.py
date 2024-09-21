#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:06:45 2021

@author: tianyu
"""


import h5py
import scipy as sp
import numpy as np
import scanpy as sc
import pandas as pd

from sklearn.neighbors import kneighbors_graph


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

def buildGraphNN(X, neighborK):
    A = kneighbors_graph(X, neighborK, mode='connectivity', metric='cosine', include_self=True)
    return A

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)



def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d



def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]))
        var = pd.DataFrame(dict_from_group(f["var"]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns

def preprocess(filename, normalize = True, sparsify = False, skip_exprs = False):
    mat, obs, var, uns = read_data(filename, sparsify = sparsify, skip_exprs = skip_exprs)
    data = mat.todense() 
    data = data[:, np.where(np.sum(data, 0) != 0)[1]]
    if len(np.where(np.sum(data, 1) == 0)[0]) != 0:
        indexes = np.where(np.sum(data, 1) != 0)[0]
        obs = obs.iloc[ indexes,:]
        data = data[indexes ,:]
    
    if normalize:
        cellTotal = np.sum(data, axis=1)
        median = np.median(cellTotal, axis=0)[0,0]
        print("median", median)
        data = data/np.sum(data, axis=1)*median
    
    return data, obs, var, uns

def prepro(filename,sparsify=False, skip_exprs=False):
    data_path = filename
    mat, obs, var, uns = read_data(data_path, sparsify=sparsify, skip_exprs=skip_exprs)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    # bach = np.array(obs["batch"])
    # bach_type, bach_label = np.unique(bach, return_inverse=True)
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label

def pre_normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    # if adata.X.size < 50e6: # check if adata.X is integer only if array is small
    #     if sp.sparse.issparse(adata.X):
    #         assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
    #     else:
    #         assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

def load_h5(filename,n_gene=None, sparsify = False, skip_exprs = False):
    
    X, Y = prepro(filename,sparsify = sparsify, skip_exprs = skip_exprs)
    # X = np.ceil(X).astype(np.int)
    count_X = X
    
    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    adata = pre_normalize(adata, copy=True, highly_genes=n_gene, size_factors=True, normalize_input=True, logtrans_input=True)
    X = adata.X.astype(np.float32)
    Y = np.array(adata.obs["Group"])
    # high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
    # count_X = count_X[:, high_variable]
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
    cluster_number = int(max(Y) - min(Y) + 1)
    
    return X, Y, count_X, size_factor, cluster_number

def preprocess_simulate_data(X,Y,n_gene=None):
    count_X = X

    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    adata = pre_normalize(adata, copy=True, highly_genes=n_gene, size_factors=True, normalize_input=True,
                          logtrans_input=True)
    X = adata.X.astype(np.float32)
    Y = np.array(adata.obs["Group"])
    # high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
    # count_X = count_X[:, high_variable]
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
    cluster_number = int(max(Y) - min(Y) + 1)

    return X, Y, count_X, size_factor, cluster_number

    
     
