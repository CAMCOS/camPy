import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sklearn.preprocessing as pp
import sklearn.cluster
from scipy.sparse.linalg.eigen.arpack import eigsh

class spectral_clustering():
    def __init__(self):
        self

def spectral_njw(affinity,n_clusters):
    affinity= affinity - scipy.sparse.diags(np.array(affinity.diagonal()).ravel(),0)
    D = scipy.sparse.diags(np.array(np.divide(1,np.sqrt(affinity.sum(axis=1)))).ravel(),dtype=np.float64)
    L = D*affinity*D
    w,v = eigsh(L,k=n_clusters,tol=1e-5)
    v = pp.normalize(v,axis=1)
    c,l,i = sklearn.cluster.k_means(v[0:,-n_clusters:-1],n_clusters)
    return l

def diffusion(affinity,n_cluster):
    V_inv = 1 / sqrt((rowSums(V[, 2:k] ^ 2)))
    V < - matrix(rep(V_inv, k - 1), ncol=k - 1) * V[, 2:k]
    V = matrix(rep(dvec_inv, k - 1), ncol=k - 1) * V
    V = (matrix(rep(lambda [2: k], each = n), ncol = k - 1) ^ t )*V