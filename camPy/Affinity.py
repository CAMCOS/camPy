from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.sparse as sparse
import sklearn.metrics as metrics
import time


def NN_graph(X,k,metric='cosine',use_values=True,f=lambda x:1-x,verbose = True):
    start_time = time.time()
    row = np.zeros(X.shape[0]*k,dtype=np.uint16)
    column = np.zeros(X.shape[0]*k,dtype=np.uint16)
    if use_values == True:
        values = np.zeros(X.shape[0]*k)
    n = 0
    elapsed_time = -20
    for i in np.arange(X.shape[0]):
        if sparse.issparse(X):
            dists = metrics.pairwise.pairwise_distances(X[i, 0:],
                                                        X,
                                                        metric=metric)
        else:
            dists = metrics.pairwise.pairwise_distances(X[i, 0:].reshape(1,-1),
                                                    X,
                                                    metric=metric)
        index = np.argpartition(dists[0,0:],k+1)
        row[n:(n+k)] = i
        column[n:(n + k)] = index[1:(k+1)]
        if use_values == True:
            values[n:(n+k)] = f(dists[0,index[1:(k+1)]])
        n += k
        if verbose:
            if time.time() -  elapsed_time - start_time > 30:
                elapsed_time = (time.time()-start_time)
                avg = (elapsed_time / i)*(X.shape[0]-i)
                print("This has taken ",
                    np.round(elapsed_time,1),
                    "seconds and is expected to take ",
                    np.round(avg,1),
                    "seconds more ")
    if use_values == True:
        NN = sparse.csr_matrix((np.array(values),
                            (np.array(row), np.array(column))),
                           dtype=np.float16)
    else:
        NN = sparse.csr_matrix((np.repeat(1, np.array(row).size),
                            (np.array(row), np.array(column))),
                           dtype=np.float16)
    return 0.5*(NN+NN.transpose())


def NN_graph_fast(X,k,metric='cosine',use_values=True,f=lambda x:1-x,verbose = True,by=100):
    start_time = time.time()
    row = np.zeros(X.shape[0]*k,dtype=np.uint16)
    column = np.zeros(X.shape[0]*k,dtype=np.uint16)
    if use_values == True:
        values = np.zeros(X.shape[0]*k)
    n = 0
    elapsed_time = -20
    for i in np.arange(start=0,stop= X.shape[0],step=by):
        dists = metrics.pairwise.pairwise_distances(X[i:(i+by), 0:],
                                                    X,
                                                    metric=metric)
        index = np.argsort(dists[0:,0:],axis=1)[0:,:k+1]
        by = dists.shape[0]
        x = np.arange(start=i,stop=(i+by))
        row[n:(n+k*by)] = np.repeat(x,k,axis=0)
        column[n:(n+k*by)] = np.array(index[0:,1:(k+1)]).ravel()
        if use_values == True:
                values[n:(n+k*by)] = f(np.array(np.sort(dists,axis=1)[0:,1:k+1]).ravel())
        n += k*by
        if verbose:
            if time.time() -  elapsed_time - start_time > 30:
                elapsed_time = (time.time()-start_time)
                avg = (elapsed_time / i)*(X.shape[0]-i)
                print("This has taken ",
                    np.round(elapsed_time,1),
                    "seconds and is expected to take ",
                    np.round(avg,1),
                    "seconds more ")
    if use_values == True:
        NN = sparse.csr_matrix((np.array(values),
                            (np.array(row), np.array(column))),
                           dtype=np.float16)
    else:
        NN = sparse.csr_matrix((np.repeat(1, np.array(row).size),
                            (np.array(row), np.array(column))),
                           dtype=np.float16)
    return 0.5*(NN+NN.transpose())

def corr_hack(data):
    # center the data by column
    centeredcolumns = data - np.repeat(data.sum(axis=0).reshape(1,-1),data.shape[0],axis=0)
    Cov_matrix = centeredcolumns*centeredcolumns.transpose()
    # calculate the length of each row (eventually scale by row&col)
    cov_rowsums = (np.power(Cov_matrix,2)).sum(axis=1)
    Corr_hack = Cov_matrix / np.sqrt(cov_rowsums)
    Similarity = (Corr_hack.transpose() / np.sqrt(cov_rowsums)).transpose()
    # scale the matrix to (0,1), excluding the diagonal
    Similarity = (Similarity - np.min(Similarity)) / (np.max(Similarity) - np.min(Similarity))
    return Similarity
