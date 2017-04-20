from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sklearn.preprocessing as pp
import sklearn.cluster
from scipy.sparse.linalg.eigen.arpack import eigsh
import sklearn.metrics as metrics

#class spectral_clustering():
#    def __init__(self):
#        self

def spectral_njw(affinity,n_clusters,compute_probs=False,metric='euclidean'):
    affinity= affinity - scipy.sparse.diags(np.array(affinity.diagonal()).ravel(),0)
    D = scipy.sparse.diags(np.array(np.divide(1,np.sqrt(affinity.sum(axis=1)))).ravel(),dtype=np.float64)
    L = D*affinity*D
    w,v = eigsh(L,k=n_clusters,tol=1e-5)
    v = pp.normalize(v,axis=1)
    clusterer = sklearn.cluster.KMeans(n_clusters)
    l= clusterer.fit_predict(v[0:,-n_clusters:-1])
    if compute_probs ==True:
        dists = metrics.pairwise.pairwise_distances(v[0:,-n_clusters:-1],clusterer.cluster_centers_,metric=metric)
        probs = dists[np.arange(l.shape[0]),l]/np.sum(dists,axis = 1)
        return l,probs**-1
    return l

def diffusion(affinity,n_clusters,alpha,t):
    affinity = affinity - scipy.sparse.diags(np.array(affinity.diagonal()).ravel(), 0)
    D_alpha = scipy.sparse.diags(np.array(np.power(affinity.sum(axis=1),-alpha)).ravel(),dtype=np.float64)
    affinity = D_alpha*affinity*D_alpha
    D = scipy.sparse.diags(np.array(np.power(affinity.sum(axis=1), -1)).ravel(), dtype=np.float64)
    L_alpha = D*affinity*D
    w, v = eigsh(L_alpha, k=n_clusters, tol=1e-5)
    v = v*np.power(w,t)
    c, l, i = sklearn.cluster.k_means(v[0:, -n_clusters:-1], n_clusters)
    return l
