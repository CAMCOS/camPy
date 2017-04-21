from __future__ import division, absolute_import, print_function

import numpy as np
import scipy.sparse
import scipy.stats
import sklearn

class preprocessing():
    """
    Possible preprocessing techniques
    """
    def __init__(self,matrix):
        self.ppmatrix = matrix
        self.index = np.ones(matrix.shape[1],dtype=np.bool)

    def remove_columns_less_than_one(self):
        temp = np.arange(self.index.shape[0])[self.index]
        test = np.array(self.ppmatrix.sum(axis=0) > 1).ravel()
        self.index[temp[test==False]] = False
        self.ppmatrix = self.ppmatrix[0:,test]

    def make_binary(self):
        self.ppmatrix = self.ppmatrix.astype(bool)

    def normalize_rows(self,type):
        self.ppmatrix = sklearn.preprocessing.normalize(self.ppmatrix, axis=1, norm=type)

    def column_weighting(self,type):
        weighter = factor_weighting(self.ppmatrix)
        self.ppmatrix = self.ppmatrix*getattr(weighter, type)()

    def remove_common_columns(self,occurance):
        col_occurance = factor_weighting(self.ppmatrix).col_occurance
        test = np.array(col_occurance < occurance).ravel()
        temp = np.arange(self.index.shape[0])[self.index]
        self.index[temp[test==False]] = False
        self.ppmatrix = self.ppmatrix[0:, np.array(col_occurance < occurance).ravel()]

    def LSA(self,n_components):
        pca = sklearn.decomposition.TruncatedSVD(n_components=n_components)
        pca.fit(self.ppmatrix)
        self.ppmatrix = pca.transform(self.ppmatrix)

    def PCA(self,n_components):
        u, s, v = scipy.sparse.linalg.svds(self.ppmatrix,k=n_components)
        self.ppmatrix = self.ppmatrix*v.transpose()

    def do_it_all(self,occurance=0.6,norm='l2',weighting='IDF',n_components = 100):
        self.remove_columns_less_than_one()
        self.remove_common_columns(occurance)
        self.make_binary()
        self.normalize_rows(norm)
        self.column_weighting(weighting)
        self.LSA(n_components=n_components)
        return self.ppmatrix

class factor_weighting():
    """
    Several weighting functions for a matrix A
    """
    def __init__(self,A):
        # A is the matrix you want to find weights for
        self.col_occurance = np.array((A>0).sum(axis=0)).flatten()/A.shape[0]

    def threashold(self,start,end):
        # Hard threashold
        # c is the number of clusters
        weights = (self.col_occurance <end)*(self.col_occurance >start)
        return scipy.sparse.diags(weights,dtype=np.bool)

    def linear(self,c,start,end):
        m  = -1/(start - 1/c)
        b = - m / c + 1
        m2 = 1/(1/c-end)
        b2 = 1 - m2/c
        def g(x):
            if x < 1/c:
                return m*x+b
            else:
                return m2*x+b2
        g = np.vectorize(g)
        return scipy.sparse.diags(g(self.col_occurance))

    def beta(self,c,b):
        a  = (b/c -2/c + 1) / (1-1/c)
        return scipy.sparse.diags(scipy.stats.beta.pdf(self.col_occurance,a,b))

    def IDF(self,power=2):
        return scipy.sparse.diags(np.power(-np.log(self.col_occurance),power))
