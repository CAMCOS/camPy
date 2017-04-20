from __future__ import division, absolute_import, print_function

import numpy as np
from matplotlib import pyplot as plt


def maximum_number_of_correctly_classified_points(num):
    index_row = np.zeros(num.shape[0],dtype=np.bool)
    index_column = np.zeros(num.shape[1],dtype=np.bool)
    assignment = np.zeros(num.shape[0],dtype=np.int)-1
    count = 0
    while count <= num.shape[0]:
        num_temp = num[np.ix_(index_row==False,index_column==False)]
        if np.sum(num_temp)==0:
            break
        rows = np.arange(num.shape[0])[index_row == False]
        columns = np.arange(num.shape[1])[index_column == False]
        row_winners, column_winners = find_clear_winners(num_temp,rows,columns)
        assignment[row_winners] = column_winners
        index_column += np.in1d(np.arange(num.shape[1]),column_winners)
        index_row +=  np.in1d(np.arange(num.shape[0]),row_winners)
        if columns ==[]:
            break
        count +=1
    n=np.sum(num[np.arange(num.shape[0])[assignment!=-1],assignment[assignment!=-1]])
    return n,assignment

def find_clear_winners(num,rows,columns):
    max_row = np.array(np.argmax(num, axis=1)).ravel()
    max_col = np.array(np.argmax(num[0:, max_row], axis=0)).ravel()
    index = max_col == np.arange(num.shape[0])
    row_winners = rows[np.arange(num.shape[0])[index]]
    column_winners = columns[max_row[index]]
    return row_winners,column_winners

def computing_percentage_of_misclassified_points(indices,trueLabels):
    K = np.unique(trueLabels).size
    planeSizes = np.zeros((K,1))
    k = 0
    for m in np.unique(trueLabels):
        planeSizes[k] = np.sum(trueLabels == m)
        k = k+1

    num = np.zeros((K,np.unique(indices).size))
    k= 0
    for n in np.unique(trueLabels):
        j=0
        for m in np.sort(np.unique(indices)):
            num[k,j] = np.sum((indices==m)*(trueLabels==n))
            j=j+1
        k = k+1
    n,assignments = maximum_number_of_correctly_classified_points(num)
    p = 1-n/sum(planeSizes)
    return p,assignments

def plot_clus(l,t):
    counts = np.zeros(np.unique(t).shape[0])
    count = np.zeros(np.unique(t).shape[0])
    ax = plt.subplot(111)
    color = plt.cm.rainbow(np.arange(np.unique(l).size) / (np.unique(l).size-1))
    o = 0
    han = list()
    for i in np.unique(l):
        count = count + counts
        counts = np.zeros(np.unique(t).shape[0])
        k = 0
        for j in np.unique(t):
            counts[k] = np.sum((l == i)[t == j])
            k = k + 1
        p1 = ax.bar(np.arange(np.unique(t).shape[0]) + 1, counts, color=color[o], bottom=count, width=.9)
        han.append(p1)
        o = o + 1
    plt.xlim((0, np.unique(t).shape[0]+1))
    ax.set_xticks(np.arange(np.unique(t).shape[0]) + .9 / 2)
    ax.set_xticklabels(np.arange(np.unique(t).shape[0])+1, rotation=90)
    plt.legend(han, np.unique(l) + 1)
    plt.show()

def knn(A,k):
    m = 0
    for i in np.arange(A.shape[0]):
        m = m+np.sort(A[i,0:])[k]
    return m/A.shape[0]