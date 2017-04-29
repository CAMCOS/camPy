from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sklearn.preprocessing as pp
import sklearn.cluster
from scipy.sparse.linalg.eigen.arpack import eigsh
import sklearn.metrics as metrics

def spectral_njw(affinity,n_clusters):
    affinity= affinity - scipy.sparse.diags(np.array(affinity.diagonal()).ravel(),0)
    affinity, remove = remove_degree_zero(affinity)
    D = scipy.sparse.diags(np.array(np.divide(1,np.sqrt(affinity.sum(axis=1)))).ravel(),dtype=np.float64)
    L = D*affinity*D
    w,v = eigsh(L,k=n_clusters,tol=1e-5)
    v = pp.normalize(v,axis=1)
    clusterer = sklearn.cluster.KMeans(n_clusters)
    l= clusterer.fit_predict(v[0:,-n_clusters:-1])
    l = np.insert(l, remove, 0)
    return l

def diffusion(affinity,n_clusters,alpha,t):
    affinity = affinity - scipy.sparse.diags(np.array(affinity.diagonal()).ravel(), 0)
    affinity,remove = remove_degree_zero(affinity)
    D_alpha = scipy.sparse.diags(np.array(np.power(affinity.sum(axis=1),-alpha)).ravel(),dtype=np.float64)
    affinity = D_alpha*affinity*D_alpha
    affinity = (affinity+affinity.transpose())/2
    w, v = eigsh(affinity, k=n_clusters, tol=1e-5)
    v = D_alpha*v
    v = v*np.power(w,t)
    v = pp.normalize(v, axis=1)
    c, l, i = sklearn.cluster.k_means(v[0:, -n_clusters:-1], n_clusters)
    l = np.insert(l,remove,0)
    return l

def spectral_ncut(affinity,n_clusters):
    affinity= affinity - scipy.sparse.diags(np.array(affinity.diagonal()).ravel(),0)
    affinity,remove = remove_degree_zero(affinity)
    D = scipy.sparse.diags(np.array(np.divide(1,affinity.sum(axis=1))).ravel(),dtype=np.float64)
    L = D*affinity
    w,v = eigsh(L,k=n_clusters,tol=1e-5)
    clusterer = sklearn.cluster.KMeans(n_clusters)
    l= clusterer.fit_predict(v[0:,-n_clusters:-1])
    l = np.insert(l,remove,0)
    return l

def remove_degree_zero(affinity):
    removed = np.arange(affinity.shape[0])[np.array(affinity.sum(axis=1) ==0).ravel()]
    if removed.shape[0] >0:
        print("Warning: there are observation with zero similarity to anything.",
              " The following observations were assign to group 0\n",removed)
        affinity = affinity[np.array(affinity.sum(axis=1) >0).ravel()]
        affinity = affinity[0:,np.array(affinity.sum(axis=0) >0).ravel()]
    return affinity,removed
