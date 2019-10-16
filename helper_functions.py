"""*****************************************************************************************
MIT License

Copyright (c) 2019 Ibrahim Jubran, Alaa Maalouf, Dan Feldman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************************"""


################################### NOTES ###########################################
# - Please cite our paper when using the code: 
#                "Accurate Coresets"
#    Ibrahim Jubran and Alaa Maalouf and Dan Feldman
#
# - Code for other coresets, both accurate and eps-coresets, will be published soon. 
#####################################################################################


import numpy as np
from scipy.linalg import lstsq
from sympy import Matrix
import random
from sklearn import linear_model
import matplotlib.pyplot as plt
import sys
from sklearn import preprocessing
import time
import math
import scipy.linalg as spla


def Caratheodory(P, u, dtype='float64'):
    """
    Implementation of the Caratheodory Theorem(1907)
    input: a numpy array P containing n rows (points), each of size d, and a positive vector of weights u (that sums to 1)
    output:a new vector of weights new_u that satisfies :
                1. new_u is positive and sums to 1
                2. new_u has at most d+1 non zero entries
                3. the weighted sum of P and u (input) is the same as the weighted sum of P and new_u (output)
    computation time: O(n^2d^2)
    """
    while 1:
        n = np.count_nonzero(u)
        d = P.shape[1]
        u_non_zero = np.nonzero(u)

        if n <= d + 1:
            return u

        A = P[u_non_zero]
        reduced_vec = np.outer(A[0], np.ones(A.shape[0]-1, dtype = dtype))
        A = A[1:].T - reduced_vec

        _, _, V = np.linalg.svd(A, full_matrices=True)
        v=V[-1]
        v = np.insert(v, [0],   -1 * np.sum(v))

        idx_good_alpha = np.nonzero(v > 0)
        alpha = np.min(u[u_non_zero][idx_good_alpha]/v[idx_good_alpha])

        w = np.zeros(u.shape[0] , dtype = dtype)
        tmp_w = u[u_non_zero] - alpha * v
        tmp_w[np.argmin(tmp_w)] = 0.0
        w[u_non_zero] = tmp_w
        w[u_non_zero][np.argmin(w[u_non_zero] )] = 0
        u = w


def Fast_Caratheodory(P,u,coreset_size, dtype = 'float64'):
    """
    Our fast and accurate implementation of Caratheodory's Theorem
    Input: a numpy array P containing n rows (points), each of size d, and a positive vector of weights u (if u does not
    sum to 1, we first normalize u by its sum, then multiply u back by its original sum before returning it)
    Output: a new vector of positive weights new_u that satisfies :
                 1. new_u has at most d+1 non zero entries
                 2. the weighted sum of P and u (input) is the same as the weighted sum of P and new_u (output)
    Computation time: O(nd+logn*d^4)
    """
    d = P.shape[1]
    n = P.shape[0]
    m = 2*d + 2
    if n <= d + 1:
        return u.reshape(-1)

    u_sum = np.sum(u)
    u = u/u_sum
    chunk_size = math.ceil(n/m)
    current_m = math.ceil(n/chunk_size)

    add_z = chunk_size - int (n%chunk_size)
    u = u.reshape(-1,1)
    if add_z != chunk_size:
        zeros = np.zeros((add_z, P.shape[1]), dtype = dtype)
        P = np.concatenate((P, zeros))
        zeros = np.zeros((add_z, u.shape[1]), dtype = dtype)
        u = np.concatenate((u, zeros))
    
    idxarray = np.array(range(P.shape[0]) )
    
    p_groups = P.reshape(current_m, chunk_size, P.shape[1])
    u_groups = u.reshape(current_m, chunk_size)
    idx_group = idxarray.reshape(current_m, chunk_size)
    u_nonzero = np.count_nonzero(u)

    if not coreset_size:
        coreset_size = d+1
    while u_nonzero > coreset_size:

        groups_means = np.einsum('ijk,ij->ik',p_groups, u_groups)
        group_weigts = np.ones(groups_means.shape[0], dtype = dtype)*1/current_m

        Cara_u_idx = Caratheodory(groups_means , group_weigts,dtype = dtype )

        IDX = np.nonzero(Cara_u_idx)

        new_P = p_groups[IDX].reshape(-1,d)

        subset_u = (current_m * u_groups[IDX] * Cara_u_idx[IDX][:, np.newaxis]).reshape(-1, 1)
        new_idx_array = idx_group[IDX].reshape(-1,1)
        ##############################################################################3
        u_nonzero = np.count_nonzero(subset_u)
        chunk_size = math.ceil(new_P.shape[0]/ m)
        current_m = math.ceil(new_P.shape[0]/ chunk_size)

        add_z = chunk_size - int(new_P.shape[0] % chunk_size)
        if add_z != chunk_size:
            new_P = np.concatenate((new_P, np.zeros((add_z, new_P.shape[1]), dtype = dtype)))
            subset_u = np.concatenate((subset_u, np.zeros((add_z, subset_u.shape[1]),dtype = dtype)))
            new_idx_array = np.concatenate((new_idx_array, np.zeros((add_z, new_idx_array.shape[1]),dtype = dtype)))
        p_groups = new_P.reshape(current_m, chunk_size, new_P.shape[1])
        u_groups = subset_u.reshape(current_m, chunk_size)
        idx_group = new_idx_array.reshape(current_m , chunk_size)
        ###########################################################

    new_u = np.zeros(n)
    subset_u = subset_u[(new_idx_array < n)]
    new_idx_array = new_idx_array[(new_idx_array < n)].reshape(-1).astype(int)
    new_u[new_idx_array] = subset_u
    return (u_sum * new_u).reshape(-1)


def train_model(Pset, clf):
    time_start = time.time()
    weighted_data = Pset.P * np.sqrt(Pset.W.reshape(-1,1))
    weighted_labels = (Pset.Y * np.sqrt(Pset.W.reshape(-1,1))).ravel()
    clf.fit(weighted_data, weighted_labels)
    time_end = time.time()

    return time_end - time_start, clf


def get_new_clf(solver, folds=3, alphas=100):
    if "linear" == solver:
        clf = linear_model.LinearRegression(fit_intercept=False)
    if "ridge" == solver:
        clf = linear_model.Ridge(fit_intercept=False)
    elif "lasso" == solver:
        clf=linear_model.Lasso(fit_intercept=False)
    elif "elastic" == solver:
        clf = linear_model.ElasticNet(fit_intercept=False)
    return clf


def test_model(Pset, clf):
    weighted_test_data = Pset.P * np.sqrt(Pset.W.reshape(-1,1))
    weighted_test_labels = Pset.Y * np.sqrt(Pset.Y.reshape(-1,1))
    score = clf.score(weighted_test_data, weighted_test_labels)
    return score