import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import sklearn.preprocessing as pp
import sklearn.cluster


def spectral_njw(affinity,n_clusters):
    affinity= affinity - scipy.sparse.diags(np.array(affinity.diagonal()).ravel(),0)
    D = np.array(np.divide(1,np.sqrt(affinity.sum(axis=1)))).ravel()
    L = np.dot(np.dot(np.diag(D),affinity),np.diag(D))
    w,v = scipy.sparse.linalg.eigsh(L,k=n_clusters)
    v = pp.normalize(v,axis=1)
    c,l,i = sklearn.cluster.k_means(v[0:,-n_clusters:-1],n_clusters)
    return l