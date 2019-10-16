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
import scipy.linalg
import time
from helper_functions import Fast_Caratheodory, train_model, get_new_clf, test_model
from sklearn import linear_model


class WeightedSet:

    def __init__(self, P, W, Y=None):
        # P is of size nXd
        # W is of size 1Xn
        # Y is of size nXk
        ##todo add checker

        self.P = np.array(P)
        if (P.ndim == 1):
            self.P = self.P.reshape(-1, 1)
        self.n = self.P.shape[0]
        self.d = self.P.shape[1]
        self.Y = Y
        self.W = np.array(W).reshape(1, -1)
        self.dtype = P.dtype

        if (self.Y is not None) and (P.shape[0] != Y.shape[0]):
            self.Y = self.Y.reshape(self.n, -1)

        self.sum_W = np.sum(self.W);  # print (self.W.shape)
        self.weighted_sum = self.P.T.dot(self.W.T)

    # p must be a 1 dimensional nparray (p.shape = (d,) )
    # w is a number
    def add_point(self, p, w, y):
        self.P = np.append(self.P, [p], axis=0)
        self.W = np.append(self.W, w)
        if not self.Y is None: self.Y = np.append(self.Y, y)
        self.n = self.n + 1
        self.sum_W = self.sum_W + w
        self.weighted_sum = self.weighted_sum + w * p


# Accurate 1 center coreset
# input: unweighted data P (nXd) on a line in R^d (ignores the weights in Pset if they exist)
# Output: row indices of the coreset points C \subseteq P
def one_center(Pset):
    if check_weights_if_ones(Pset) == 0:
        return Pset

    P = Pset.P
    # Subtract mean and then compute the best line that fits the points
    P = P - np.mean(P, axis=0)
    U, D, Vt = scipy.linalg.svd(P, full_matrices=True)

    SSD = np.sum(D[1:])
    if (SSD > small_number):
        print("Data is not on a line!")
        return P

    v = Vt[0, :]  # The direction vector of the line that the points lie on

    # Project points on v and return the 2 points that yield the smallest and largest projection values
    P_proj = np.matmul(P, v)
    P_min_idx = np.argmin(P_proj)
    P_max_idx = np.argmax(P_proj)

    C = Pset.P[(P_min_idx, P_max_idx), :]
    Cset = WeightedSet(C, np.ones(C.shape[0]))

    return Cset


def get_normalized_weighted_set(Pset):
    W_normalized = Pset.W / Pset.sum_W
    return WeightedSet(Pset.P, W_normalized)


# vectors sum coreset
# Input: a weighted set in R^d
def vectors_sum_1(Pset):
    P_normalized = get_normalized_weighted_set(Pset)
    Cset = WeightedSet(P_normalized.weighted_sum, Pset.sum_W)
    return Cset


# Vectors sum coreset 3
# Input: a weighted set in R^d. The sum of weights is not necesarrily 1. Therefore, we first divide the input weights by
# their sum, compute the subset coreset of bounded weights using Caratheodory's theorem, then multiply the obtained
# weights by the original sum of weights.
def vectors_sum_3(Pset):
    pre_norm_sum_W = Pset.sum_W
    pre_norm_weighted_sum = Pset.weighted_sum
    Pset = get_normalized_weighted_set(Pset)

    # Add another dimension of 1's to the points of Pset
    Q = Pset.P
    Q = np.concatenate((Q, np.ones((Q.shape[0], 1))), axis=1)

    # Create another weighted set with Q and the same weights of Pset
    Qset = WeightedSet(Q, Pset.W)

    # Compute the weighted sum of Qset as a convex combination of at most d+2 points of Qset. d is the dimension of P
    # p, res, rnk, s = lstsq(Qset.P.transpose(), Qset.weighted_sum)
    W_cara  = Fast_Caratheodory(Qset.P, Qset.W, Qset.d + 1)
    C_idx = np.nonzero(W_cara)
    W_cara = W_cara[C_idx]

    # Output coreset: the points of P that correspond to the points of Q that were chosen in the function updated_cara
    Cset = WeightedSet(Pset.P[C_idx], W_cara * pre_norm_sum_W)

    # Check that the weighted coreset points = weighted input points
    if (np.linalg.norm(Cset.weighted_sum - pre_norm_weighted_sum) > small_number):
        print("Bad result - wrong weighted sum!!");

    # check that the sum of weights of the coreset = sum of weights of the input points
    if (abs(Cset.sum_W - pre_norm_sum_W) > small_number):
        print("Bad result - wrong sum of weights")
    return Cset


# 1 mean coreset 3
# Input: a weighted set in R^d. The sum of weights is not necesarrily 1. Therefore, we first divide the input weights by
# their sum, compute the subset coreset of bounded weights using Caratheodory's theorem, then multiply the obtained
# weights by the original sum of weights.
def one_mean_3(Pset):
    pre_norm_sum_W = Pset.sum_W
    pre_norm_weighted_sum = Pset.weighted_sum
    Pset = get_normalized_weighted_set(Pset)

    # Add 2 more dimensions to each point: p -> (p, ||p||, 1)
    Q = Pset.P
    Q_norms = np.linalg.norm(Q, axis=1)
    Q = np.concatenate((Q, Q_norms.reshape(Q_norms.shape[0], 1), np.ones((Q.shape[0], 1))), axis=1)

    # Create another weighted set with Q and the same weights of Pset
    Qset = WeightedSet(Q, Pset.W)

    # Compute the weighted sum of Qset as a convex combination of at most d+3 points from Qset. d is the dimension of P
    W_cara  = Fast_Caratheodory(Qset.P, Qset.W, Qset.d + 1)
    C_idx = np.nonzero(W_cara)
    P_cara = Qset.P[C_idx]
    W_cara = W_cara[C_idx]

    # testing
    Tset = WeightedSet(P_cara, W_cara)
    if (np.linalg.norm(Tset.weighted_sum - Qset.weighted_sum) > small_number):
        print("Bad coreset!!")
        return Pset

    # Output coreset: the points of P that correspond to the points of Q that were chosen in the function updated_cara
    Cset = WeightedSet(Pset.P[C_idx], W_cara * pre_norm_sum_W)

    # Check that the weighted coreset points = weighted input points
    if (np.linalg.norm(Cset.weighted_sum - pre_norm_weighted_sum) > small_number):
        print("Bad result - wrong weighted sum!!")
        return Pset

    # check that the sum of weights of the coreset = sum of weights of the input points
    if (abs(Cset.sum_W - pre_norm_sum_W) > small_number):
        print("Bad result - wrong sum of weights")
        return Pset

    return Cset



# 1 segment coreset
# Input: a set P of R^{d+1}, where P = {(t_i | p_i)}_{i=1}^n and non-negative weights W
def one_segment(Pset):
    P = Pset.P
    W = Pset.W
    sqrt_W = np.sqrt(W)
    d = Pset.d - 1  # Dimension of the points p_i
    X_unweighted = np.concatenate((np.ones((P.shape[0], 1)), P), axis=1)
    X = (X_unweighted.transpose()*sqrt_W).transpose() # Multiply the i'th row of X_unweighted by the i'th entry of sqrt_W
    U, D, Vt = scipy.linalg.svd(X, full_matrices=False)
    D = np.diag(D)

    # y is the leftmost column of DV^T
    u = np.matmul(D, Vt)[:, 0]

    c = (np.linalg.norm(u) ** 2) / (d + 2)

    w_vec = np.sqrt(c) * np.ones(d + 2)

    # compute a matrix Y such that Yu = w_vec
    Y = align_vectors(u, w_vec)

    # B is the (d+1) rightmost columns of YDV^T/sqrt(c)
    B = (np.matmul(Y, np.matmul(D, Vt)) / np.sqrt(c))[:, 1:]

    # the coreset is the rows of B
    Cset = WeightedSet(B, c*np.ones(B.shape[0]))
    one_segment_coreset_checker(Pset, Cset)

    return Cset, c

def one_segment_coreset_checker(Pset, Cset):
    d = Pset.d - 1
    a = np.random.rand(1, d)
    b = np.random.rand(1, d)

    sum_all_coreset = 0
    sum_all = 0
    for i in range(Pset.n):
        sum_all += Pset.W[0,i]*np.linalg.norm(a + b * Pset.P[i, 0] - Pset.P[i, 1:]) ** 2
    for i in range(Cset.n):
        sum_all_coreset += Cset.W[0,i]*np.linalg.norm(a + b * Cset.P[i, 0] - Cset.P[i, 1:]) ** 2
    if np.abs(sum_all_coreset - sum_all) > small_number:
        print("Bad Coreset, {} - {}".format(sum_all_coreset, sum_all))

# Input: u,v \in R^d
# Find a rotation matrix R such that Ru/||u|| = v/||v||
def align_vectors(u, v):
    d = u.size

    u = np.divide(u, np.linalg.norm(u))
    v = np.divide(v, np.linalg.norm(v))

    u_bot = orthogonal_complement(u)
    v_bot = orthogonal_complement(v)

    # Rotation matrices that align u,v with the x axis
    R_u = np.concatenate((u.reshape(1, d), u_bot.transpose()), axis=0)
    R_v = np.concatenate((v.reshape(1, d), v_bot.transpose()), axis=0)

    # align u and v by first aligning u with the x axis, then aligning the x axis with v
    R_uv = np.matmul(R_v.transpose(), R_u)

    return R_uv


# Given a unit vector x, compute the orthogonal complement of x
def orthogonal_complement(x, threshold=1e-15):
    """Compute orthogonal complement of a matrix

    this works along axis zero, i.e. rank == column rank,
    or number of rows > column rank
    otherwise orthogonal complement is empty

    TODO possibly: use normalize='top' or 'bottom'

    """

    if (abs(np.linalg.norm(x) - 1) > small_number):
        x = np.divide(x, np.linalg.norm(x))

    if (x.shape[0] == x.size):
        x = x.reshape(x.size, 1)

    # x = np.asarray(x)
    r, c = x.shape
    if r < c:
        import warnings
        warnings.warn('fewer rows than columns', UserWarning)

    # we assume svd is ordered by decreasing singular value, o.w. need sort
    s, v, d = scipy.linalg.svd(x)
    rank = (v > threshold).sum()

    oc = s[:, rank:]

    return oc


# Check if the weights of a weighted set are all ones
def check_weights_if_ones(Pset):
    if (np.sum(Pset.W - np.ones(Pset.n)) > small_number):
        print("Weights should all be ones!")
        return 0
    return 1


# Check if the sum(weights) = 1
def check_sum_weights_if_ones(Pset):
    if abs(Pset.sum_W - 1) > small_number:
        print("Weights do not sum to one! exiting..")
        return 0
    return 1


# Matrix 2 norm coreset
# Input: An unweighted set P
def matrix_norm(Pset):
    if check_weights_if_ones(Pset) == 0:
        return Pset

    P = Pset.P

    U, C = np.linalg.qr(P)

    Cset = WeightedSet(C, np.ones(C.shape[0]))

    return Cset



# Matrix 2 norm coreset which is a subset of the data
# Input: A set P with with positive weights W
def matrix_norm2(Pset, coreset_size=None):
    P = Pset.P
    Y = Pset.Y
    if Pset.Y is not None:
        P_tag = np.append(P, Y, axis=1)
    else:
        P_tag = P
    W = Pset.W
    # reshape so we can use the einsum
    n_tag = P_tag.shape[0];
    d_tag = P_tag.shape[1]
    P_tag = P_tag.reshape(n_tag, d_tag, 1);

    # build the tensor off all covariance matrices of the row vector of P_tag
    P_tag = np.einsum("ikj,ijk->ijk", P_tag, P_tag)
    P_tag = P_tag.reshape(n_tag, -1)

    coreset_weigts = Fast_Caratheodory(P_tag, W, coreset_size, dtype=Pset.dtype)
    new_idx_array = np.nonzero(coreset_weigts)

    coreset_weigts = coreset_weigts[new_idx_array]
    coreset_weigts = coreset_weigts.reshape(-1,1)

    if Pset.Y is not None:
        Cset = WeightedSet(P[new_idx_array], coreset_weigts.reshape(1, -1), Y[new_idx_array])
    else:
        Cset = WeightedSet(P[new_idx_array], coreset_weigts.reshape(1, -1), None)

    #matrix_norm_checker(Pset, Cset)
    return Cset


def matrix_norm_checker(Pset, Cset):
    x = np.random.rand(3, 1)
    weighted_C = np.diag(np.sqrt(Cset.W).reshape(-1)).dot(Cset.P)
    real_data = np.abs(np.linalg.norm(np.matmul(Pset.P, x)))
    coreset_data = np.abs(np.linalg.norm(np.matmul(weighted_C, x)))

    if (np.abs(real_data - coreset_data) > small_number):
        print("Not good!", np.abs(real_data - coreset_data))


def LMS_solvers(Pset, solver,coreset_size=None):
    Cset = matrix_norm2(Pset, coreset_size)
    if solver in ["linear", "ridge"] :
        return Cset
    elif solver in ["lasso", "elastic"]:
        Cset = WeightedSet(Cset.P, (Cset.n / Pset.n) * Cset.W,Cset.Y)
        return Cset
    else :
        print ("wrong solver name")

def coreset_train_model(Pset, clf, solver):
    t0 =time.time()
    Cset = LMS_solvers(Pset, solver, coreset_size=None)
    X = Cset.P * np.sqrt(Cset.W.reshape(-1,1))
    Y = Cset.Y * np.sqrt(Cset.W.reshape(-1,1))
    clf.fit(X,Y)
    t1 = time.time()
    return t1 - t0, clf


def regressions_checker():
    n = 240000
    d = 3
    data_range = 100
    num_of_alphas = 300
    folds = 3
    data = np.floor(np.random.rand(n, d) * data_range)
    labels = np.floor(np.random.rand(n, 1) * data_range)
    weights = np.ones(n)

    Pset = WeightedSet(data, weights, labels)
    for solver in ["lasso", "ridge", "elastic"]:
        #########RIDGE REGRESSION#############
        clf = get_new_clf(solver)
        time_coreset, clf_coreset = coreset_train_model(Pset, clf, solver=solver)
        score_coreset = test_model(Pset, clf)

        clf = get_new_clf(solver)
        time_real, clf_real = train_model(Pset, clf)
        score_real = test_model(Pset, clf)
        """   
        print (" solver: {}\nscore_diff = {}\n---->coef diff = {}\n---->coreset_time = {}\n---->data time = {}".format(
            solver,
            np.abs(score_coreset - score_real),
            np.sum(np.abs(clf_real.coef_ - clf_coreset.coef_)),
            time_coreset,
            time_real))
        """
        if np.abs(score_coreset - score_real) > small_number :
            print ("Not good. Error in LMS CORESET")

def main():
    global small_number
    small_number = 0.000001
    regressions_checker()
    P = np.array([[1.7, 0, 2], [3, 1.5, 0], [-7, 5, 0], [6, 2, 1], [1, 1, 2.2], [2, 5, 1], [6, 5, 2]])
    W = np.array([1, 2, 3, 1, 2, 1.5, 0.5])
    W = np.divide(W, np.sum(W))
    Pset = WeightedSet(P, W)
    Qset = WeightedSet(np.array([1, 2, 3]), [5, 5, 5])

    # ########## test one_center
    R, _ = np.linalg.qr(np.random.randn(3, 3))
    P_one_center = np.array([[1, 0, 0], [3, 0, 0], [-7, 0, 0], [6, 0, 0], [-1, 0, 0], [20, 0, 0]])
    P_one_center = np.matmul(P_one_center, R)
    Pset_one_center = WeightedSet(P_one_center, np.ones(P_one_center.shape[0]))
    Cset = one_center(Pset_one_center)

    # ########## test vectors_sum_1
    Cset = vectors_sum_1(Qset)

    ########## test vectors_sum_3
    vectors_sum_3(Pset)

    ########## test one_mean_3
    one_mean_3(Pset)

    ########## test one_segment
    P_one_segment = np.array([[1, 0, 2], [2, 1.5, 0], [3, 5, 0], [4, 2, 1], [5, 1, 2.2], [6, 5, 1], [7, 5, 2]])
    W_one_segment = np.random.rand(P.shape[0])
    Pset_one_segment = WeightedSet(P_one_segment, W_one_segment)
    Cset, c = one_segment(Pset_one_segment)

    ########## test matrix 2 norm
    P = np.floor(np.random.rand(10, 3) * 10)
    W_matix_norm = np.ones(P.shape[0])
    P_matrix_norm = WeightedSet(P, W_matix_norm)
    Cset = matrix_norm2(P_matrix_norm)

    matrix_norm_checker(P_matrix_norm, Cset)

    print("All good!")


if __name__ == '__main__':
    main()

