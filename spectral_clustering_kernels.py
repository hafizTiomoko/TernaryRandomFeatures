import pylab as pl
import numpy as np
from time import time
import scipy.sparse as sp

# Import datasets, classifiers and performance metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler,
                                          Nystroem)
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted
#from sklearn.utils.validation import _deprecate_positional_args
from sklearn.random_projection import SparseRandomProjection

from scipy.optimize import least_squares
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression
#from memory_profiler import profile
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import normalize, StandardScaler
from numpy import linalg as la
import scipy
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

pi = np.pi

class RFFSampler(TransformerMixin, BaseEstimator):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.
    It implements a variant of Random Kitchen Sinks.[1]
    Read more in the :ref:`User Guide <rbf_kernel_approx>`.
    Parameters
    ----------
    gamma : float, default=1.0
        Parameter of RBF kernel: exp(-gamma * x^2)
    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    random_offset_ : ndarray of shape (n_components,), dtype=float64
        Random offset used to compute the projection in the `n_components`
        dimensions of the feature space.
    random_weights_ : ndarray of shape (n_features, n_components),\
        dtype=float64
        Random projection directions drawn from the Fourier transform
        of the RBF kernel.
    Examples
    --------
    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSampler(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=5)
    >>> clf.score(X_features, y)
    1.0
    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.
    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)
    """
    #@_deprecate_positional_args
    def __init__(self, *, n_components, gamma=1., random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.
        Samples random projection according to n_features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the transformer.
        """

        X = self._validate_data(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        self.random_weights_ = (np.sqrt(2 * self.gamma) * random_state.normal(
            size=(n_features, self.n_components)))
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse='csr', reset=False)
        projection = safe_sparse_dot(X, self.random_weights_)
        #projection = np.cos(projection) #np.cos(projection) #+
        projection = np.maximum(projection, 0)
        #projection *= np.sqrt(2.) / np.sqrt(self.n_components)
        return projection

class SparseSampler(TransformerMixin, BaseEstimator):
    """Approximates feature map of an RBF kernel by Monte Carlo approximation
    of its Fourier transform.
    It implements a variant of Random Kitchen Sinks.[1]
    Read more in the :ref:`User Guide <rbf_kernel_approx>`.
    Parameters
    ----------
    gamma : float, default=1.0
        Parameter of RBF kernel: exp(-gamma * x^2)
    n_components : int, default=100
        Number of Monte Carlo samples per original feature.
        Equals the dimensionality of the computed feature space.
    random_state : int, RandomState instance or None, default=None
        Pseudo-random number generator to control the generation of the random
        weights and random offset when fitting the training data.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    random_offset_ : ndarray of shape (n_components,), dtype=float64
        Random offset used to compute the projection in the `n_components`
        dimensions of the feature space.
    random_weights_ : ndarray of shape (n_features, n_components),\
        dtype=float64
        Random projection directions drawn from the Fourier transform
        of the RBF kernel.
    Examples
    --------
    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSampler(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier(max_iter=5, tol=1e-3)
    >>> clf.fit(X_features, y)
    SGDClassifier(max_iter=5)
    >>> clf.score(X_features, y)
    1.0
    Notes
    -----
    See "Random Features for Large-Scale Kernel Machines" by A. Rahimi and
    Benjamin Recht.
    [1] "Weighted Sums of Random Kitchen Sinks: Replacing
    minimization with randomization in learning" by A. Rahimi and
    Benjamin Recht.
    (https://people.eecs.berkeley.edu/~brecht/papers/08.rah.rec.nips.pdf)
    """
    #@_deprecate_positional_args
    def __init__(self, *, s_plus, s_minus, n_components, gamma=1., random_state=None):
        self.s_plus = s_plus
        self.s_minus = s_minus
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.
        Samples random projection according to n_features.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        self : object
            Returns the transformer.
        """

        X = self._validate_data(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        #self.random_weights_ = (np.sqrt(2 * self.gamma) * random_state.normal(
        #    size=(n_features, self.n_components)))
        self.random_weights_ = _sparse_random_matrix(n_features, self.n_components, density='auto',
                              random_state=random_state)

        self.random_offset_ = random_state.uniform(0, 2 * np.pi,
                                                   size=self.n_components)
        return self

    def transform(self, X):
        """Apply the approximate feature map to X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse='csr', reset=False)
        projection = safe_sparse_dot(X, self.random_weights_)
        #transformer = SparseRandomProjection(n_components=self.n_components, density=0.001)

        #projection = transformer.fit_transform(X)
        #projection += self.random_offset_
        projection = (projection > (np.sqrt(2) * self.s_plus)).astype(float) - (
                    projection < (np.sqrt(2) * self.s_minus)).astype(float)
        #projection = (np.sign(projection-(np.sqrt(2) * self.s_plus)) + np.sign(projection-(np.sqrt(2) * self.s_minus)))/2
        #np.cos(projection, projection)
        projection *= np.sqrt(2.) / np.sqrt(self.n_components)
        return projection

def compute_thresholds(tau):
    F = lambda x: ((np.exp(-x[0] ** 2 / tau) + np.exp(-x[1] ** 2 / tau)) / np.sqrt(2 * pi * tau) - np.exp(-tau / 2),
                   (-x[0] * np.exp(-x[0] ** 2 / tau) + x[1] * np.exp(-x[1] ** 2 / tau)) / (
                       np.sqrt(2 * pi * tau ** 3)) - np.exp(-tau / 2) / 2)
    res = least_squares(F, (1, 1), bounds=((0, 0), (1, 1)))
    return res.x

def _check_density(density, n_features):
    """Factorize density check according to Li et al."""
    if density == 'auto':
        density = 1 / np.sqrt(n_features)

    elif density <= 0 or density > 1:
        raise ValueError("Expected density in range ]0, 1], got: %r"
                         % density)
    return density


def _check_input_size(n_components, n_features):
    """Factorize argument checking for random matrix generation."""
    if n_components <= 0:
        raise ValueError("n_components must be strictly positive, got %d" %
                         n_components)
    if n_features <= 0:
        raise ValueError("n_features must be strictly positive, got %d" %
                         n_features)

def _sparse_random_matrix(n_components, n_features, density='auto',
                          random_state=None):
    """Generalized Achlioptas random sparse matrix for random projection.
    Setting density to 1 / 3 will yield the original matrix by Dimitris
    Achlioptas while setting a lower value will yield the generalization
    by Ping Li et al.
    If we note :math:`s = 1 / density`, the components of the random matrix are
    drawn from:
      - -sqrt(s) / sqrt(n_components)   with probability 1 / 2s
      -  0                              with probability 1 - 1 / s
      - +sqrt(s) / sqrt(n_components)   with probability 1 / 2s
    Read more in the :ref:`User Guide <sparse_random_matrix>`.
    Parameters
    ----------
    n_components : int,
        Dimensionality of the target projection space.
    n_features : int,
        Dimensionality of the original source space.
    density : float or 'auto', default='auto'
        Ratio of non-zero component in the random projection matrix in the
        range `(0, 1]`
        If density = 'auto', the value is set to the minimum density
        as recommended by Ping Li et al.: 1 / sqrt(n_features).
        Use density = 1 / 3.0 if you want to reproduce the results from
        Achlioptas, 2001.
    random_state : int, RandomState instance or None, default=None
        Controls the pseudo random number generator used to generate the matrix
        at fit time.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    components : {ndarray, sparse matrix} of shape (n_components, n_features)
        The generated Gaussian random matrix. Sparse matrix will be of CSR
        format.
    See Also
    --------
    SparseRandomProjection
    References
    ----------
    .. [1] Ping Li, T. Hastie and K. W. Church, 2006,
           "Very Sparse Random Projections".
           https://web.stanford.edu/~hastie/Papers/Ping/KDD06_rp.pdf
    .. [2] D. Achlioptas, 2001, "Database-friendly random projections",
           http://www.cs.ucsc.edu/~optas/papers/jl.pdf
    """
    #_check_input_size(n_components, n_features)
    density = _check_density(density, n_features)
    rng = check_random_state(random_state)

    if density == 1:
        # skip index generation if totally dense
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return 1 / np.sqrt(n_components) * components

    else:
        # Generate location of non zero elements
        indices = []
        offset = 0
        indptr = [offset]
        for _ in range(n_components):
            # find the indices of the non-zero components for row i
            n_nonzero_i = rng.binomial(n_features, density)
            indices_i = sample_without_replacement(n_features, n_nonzero_i,
                                                   random_state=rng)
            indices.append(indices_i)
            offset += n_nonzero_i
            indptr.append(offset)

        indices = np.concatenate(indices)

        # Among non zero components the probability of the sign is 50%/50%
        data = rng.binomial(1, 0.5, size=np.size(indices)) * 2 - 1

        # build the CSR structure by concatenating the rows
        components = sp.csr_matrix((data, indices, indptr),
                                   shape=(n_components, n_features))

        return np.sqrt(1 / density) / np.sqrt(n_components) * components

def gen_data(testcase, selected_target, T, p, cs, means=None, covs=None):
    rng = np.random

    if testcase is 'mnist':
        (data, labels), _ = mnist.load_data()
        data = data.reshape(-1,784)
        #mnist = fetch_mldata('MNIST original')
        #data, labels = mnist.data, mnist.target

        # feel free to choose the number you like :)
        selected_target = [6, 8]
        p = 784
        K = len(selected_target)

        # get the whole set of selected number
        data_full = []
        data_full_matrix = np.array([]).reshape(p, 0)
        ind = 0
        for i in selected_target:
            locate_target_train = np.where(labels == i)[0]
            data_full.append(data[locate_target_train].T)
            data_full_matrix = np.concatenate((data_full_matrix, data[locate_target_train].T), axis=1)
            ind += 1

        # recentering and normalization to satisfy Assumption 1 and
        T_full = data_full_matrix.shape[1]
        mean_selected_data = np.mean(data_full_matrix, axis=1).reshape(p, 1)
        norm2_selected_data = np.sum((data_full_matrix - np.mean(data_full_matrix, axis=1).reshape(p, 1)) ** 2,
                                     (0, 1)) / T_full
        for i in range(K):
            data_full[i] = data_full[i] - mean_selected_data
            data_full[i] = data_full[i] * np.sqrt(p) / np.sqrt(norm2_selected_data)

        # get the statistics of MNIST data
        means = []
        covs = []
        for i in range(K):
            data_tmp = data_full[i]
            T_tmp = data_tmp.shape[1]
            means.append(np.mean(data_tmp, axis=1).reshape(p, 1))
            covs.append((data_tmp @ (data_tmp.T) / T_tmp - means[i] @ (means[i].T)).reshape(p, p))

        X = np.array([]).reshape(p, 0)
        Omega = np.array([]).reshape(p, 0)
        y = []

        ind = 0
        for i in range(K):
            data_tmp = data_full[i]
            X = np.concatenate((X, data_tmp[:, range(np.int(cs[ind] * T))]), axis=1)
            Omega = np.concatenate((Omega, data_tmp[:, range(np.int(cs[ind] * T))] - np.outer(means[ind], np.ones(
                (1, np.int(T * cs[ind]))))), axis=1)
            y = np.concatenate((y, 2 * (ind - K / 2 + .5) * np.ones(np.int(T * cs[ind]))))
            ind += 1

    elif testcase is 'cifar10':
        data_full = np.load('./cifar/cifar10_embeddings.npz')
        data = data_full['x_train']
        labels = data_full['y_train']
        labels = np.argmax(labels, axis = 1)
        #data_t = data_full['x_test']
        #target_t = data_full['y_test']

        #(data, labels), _ = cifar100.load_data()
        data = data.reshape(-1,2048)

        # feel free to choose the number you like :)
        p = 2048
        K = len(selected_target)

        # get the whole set of selected number
        data_full = []
        data_full_matrix = np.array([]).reshape(p, 0)
        ind = 0
        for i in selected_target:
            locate_target_train = np.where(labels == i)[0]
            data_full.append(data[locate_target_train].T)
            data_full_matrix = np.concatenate((data_full_matrix, data[locate_target_train].T), axis=1)
            ind += 1

        # recentering and normalization to satisfy Assumption 1 and
        T_full = data_full_matrix.shape[1]
        mean_selected_data = np.mean(data_full_matrix, axis=1).reshape(p, 1)
        norm2_selected_data = np.sum((data_full_matrix - np.mean(data_full_matrix, axis=1).reshape(p, 1)) ** 2,
                                     (0, 1)) / T_full
        for i in range(K):
            data_full[i] = data_full[i] - mean_selected_data
            data_full[i] = data_full[i] * np.sqrt(p) / np.sqrt(norm2_selected_data)

        # get the statistics of MNIST data
        means = []
        covs = []
        for i in range(K):
            data_tmp = data_full[i]
            T_tmp = data_tmp.shape[1]
            means.append(np.mean(data_tmp, axis=1).reshape(p, 1))
            covs.append((data_tmp @ (data_tmp.T) / T_tmp - means[i] @ (means[i].T)).reshape(p, p))

        X = np.array([]).reshape(p, 0)
        Omega = np.array([]).reshape(p, 0)
        y = []

        ind = 0
        for i in range(K):
            data_tmp = data_full[i]
            print(np.shape(data_tmp))
            X = np.concatenate((X, data_tmp[:, range(np.int(cs[ind] * T))]), axis=1)
            Omega = np.concatenate((Omega, data_tmp[:, range(np.int(cs[ind] * T))] - np.outer(means[ind], np.ones(
                (1, np.int(T * cs[ind]))))), axis=1)
            y = np.concatenate((y, 2 * (ind - K / 2 + .5) * np.ones(np.int(T * cs[ind]))))
            ind += 1
    else:
        X = np.array([]).reshape(p, 0)
        Omega = np.array([]).reshape(p, 0)
        y = []

        K = len(cs)
        for i in range(K):
            tmp = rng.multivariate_normal(means[i], covs[i], size=np.int(T * cs[i])).T
            X = np.concatenate((X, tmp), axis=1)
            Omega = np.concatenate((Omega, tmp - np.outer(means[i], np.ones((1, np.int(T * cs[i]))))), axis=1)
            y = np.concatenate((y, 2 * (i - K / 2 + .5) * np.ones(np.int(T * cs[i]))))

    X = X / np.sqrt(p)
    Omega = Omega / np.sqrt(p)

    return X, Omega, y, means, covs


def estim_tau(X):
    tau = np.mean(np.diag(X.T @ X))

    return tau
########################## UTILS FUNCTIONS #####################
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

####################################
# The digits dataset
#digits = datasets.fetch_openml("MNIST original")
#(data, target), _ = mnist.load_data()
testcase = 'GMM'  ### 'mnist' , 'cifar10' , 'fashion-mnist', 'imdb' , 'boston'

if testcase is 'mnist':
    (data, target), (data_t, target_t) = mnist.load_data()
    sample_sizes = 70 * np.arange(1, 10)
elif testcase is 'fashion':
    (data, target), (data_t, target_t) = fashion_mnist.load_data()
    sample_sizes = 70 * np.arange(1, 10)
elif testcase is 'cifar10':
    # (data, target), (data_t, target_t) = cifar10.load_data()
    data_full = np.load('./cifar/cifar10_embeddings.npz')
    data = data_full['x_train']
    target = data_full['y_train']
    data_t = data_full['x_test']
    target_t = data_full['y_test']
    sample_sizes = 10 * np.arange(1, 200, 10)

elif testcase is 'imagenet':
    # (data, target), (data_t, target_t) = cifar10.load_data()
    data = np.load('./imagenet/features_imagenet_dnet.npz')
    target = np.load('./imagenet/labels_imagenet_dnet.npz')
    data_t = []
    target_t = []
    sample_sizes = 10 * np.arange(1, 200, 10)
elif testcase is 'cifar100':
    (data, target), (data_t, target_t) = cifar100.load_data()
    sample_sizes = 10 * np.arange(1, 200, 10)
elif testcase is 'svhn':
    data_full = np.load('./svhn/svhn_embeddings.npz')
    data = data_full['x_train']
    target = data_full['y_train']
    data_t = data_full['x_test']
    target_t = data_full['y_test']
    sample_sizes = 10 * np.arange(1, 200, 10)
elif testcase is 'GMM':
    print('GMM')
    sample_sizes = 70 * np.arange(1, 10)

    n = 1024
    n_test = 1024
    p = 784
    K = 4
    cs = 1 / K * np.ones((K, 1))
    means = []
    covs = []
    test_case = 'imagenet'
    if test_case is 'iid':
        for i in range(K):
            means.append(np.zeros(p))
            covs.append(np.eye(p))
    elif test_case is 'means':
        for i in range(K):
            means.append(np.concatenate((np.zeros(i), 4 * np.ones(1), np.zeros(p - i - 1))))
            covs.append(np.eye(p))
    elif test_case is 'var':
        for i in range(K):
            means.append(np.zeros(p))
            covs.append(np.eye(p) * (1 + 8 * i / np.sqrt(p)))
    elif test_case is 'orth':
        for i in range(K):
            means.append(np.zeros(p))
            covs.append(np.diag(np.concatenate((
                np.ones(np.int(np.sum(cs[0:i] * p))), 4 * np.ones(np.int(cs[i] * p)),
                np.ones(np.int(np.sum(cs[i + 1:] * p)))))))
    elif test_case is 'mixed':
        for i in range(K):
            means.append(np.concatenate((np.zeros(i), 4 * np.ones(1), np.zeros(p - i - 1))))
            covs.append((1 + 4 * i / np.sqrt(p)) * scipy.linalg.toeplitz([(.4 * i) ** x for x in range(p)]))
            selected_target = []
    elif test_case is 'mnist':
        n = 1024
        p = 784
        selected_target = [6, 8]

    elif test_case is 'cifar10' or testcase is 'imagenet':
        n = 1024
        p = 2048
        selected_target = [1, 2, 3, 4]

    X, Omega, y, _, _ = gen_data(test_case, selected_target, n, p, cs, means, covs)
    X_test, Omega_test, y_test, _, _ = gen_data(test_case, selected_target, n_test, p, cs, means, covs)

    data = X.T / np.sqrt(p)
    target = y
    data_t = X_test.T / np.sqrt(p)
    target_t = y_test

    # (data, target), (data_t, target_t) = cifar100.load_data()
# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
# data = data / 255.

if testcase is 'mnist':
    data = data.reshape(-1, 784)
    data_t = data_t.reshape(-1, 784)
elif testcase is 'fashion':
    data = data.reshape(-1, 784)
    data_t = data_t.reshape(-1, 784)
elif testcase is 'cifar10' or testcase is 'imagenet':
    # data = data.reshape(-1,32*32*3)
    # data_t = data_t.reshape(-1, 32 * 32 * 3)
    data = data.reshape(-1, 2048)
    data_t = data_t.reshape(-1, 2048)
elif testcase is 'cifar100':
    data = data.reshape(-1, 32 * 32 * 3)
    data_t = data_t.reshape(-1, 32 * 32 * 3)
elif testcase is 'svhn':
    data = data.reshape(-1, 2048)
    data_t = data_t.reshape(-1, 2048)

data, target = shuffle(data, target)

n = 1024
data_train, targets_train = data[:n], target[:n]
data_test, targets_test = data_t[:n], target_t[:n]
if testcase is 'cifar10' or testcase is 'svhn' or testcase is 'imagenet':
    targets_train = np.argmax(targets_train, axis=1)
    targets_test = np.argmax(targets_test, axis=1)

if testcase is 'mnist':
    tau_est = np.trace(data_train.T @ data_train) / n / 784
elif testcase is 'fashion':
    tau_est = np.trace(data_train.T @ data_train) / n / 784
elif testcase is 'cifar10':
    # tau_est = np.trace(data_train.T @ data_train) / n / (32*32*3)
    tau_est = np.trace(data_train.T @ data_train) / n / (2048)
elif testcase is 'cifar100':
    tau_est = np.trace(data_train.T @ data_train) / n / (32 * 32 * 3)
elif testcase is 'svhn':
    tau_est = np.trace(data_train.T @ data_train) / n / (2048)
elif testcase is 'GMM':
    tau_est = estim_tau(X)

print(tau_est)
thresholds = compute_thresholds(tau_est)
s_minus = np.min(thresholds)
s_plus = np.max(thresholds)
print(s_minus, s_plus)




#linear_reg_time = time()
#reg.fit(data_train, targets_train)
#linear_reg_score = reg.score(data_test, targets_test)
#linear_reg_time = time() - linear_reg_time


#sample_sizes = [5, 10, 50, 100, 500, 1000]
samples = []
fourier_scores = []
nystroem_scores = []
sparse_scores = []
fourier_times = []
nystroem_times = []
sparse_times = []
linear_scores = []
kernel_scores = []
linear_times = []
kernel_times = []
n_classes = len(np.unique(targets_train))

def main():
    for D in sample_sizes:
        samples.append(D)
        start = time()
        feature_map_fourier = RFFSampler(gamma=.031, random_state=1, n_components=D)
        WX = feature_map_fourier.fit_transform(data_train)
        K = WX @ np.transpose(WX)
        #K = nearestPD(K)
        #print(isPD(K))
        U_Phi_c,D_Phi_c,_ = np.linalg.svd(K)
        U_Phi_c = U_Phi_c[:,:n_classes+1]
        #estimator = make_pipeline(StandardScaler(), KMeans(n_clusters=n_classes, random_state=0)).fit(U_Phi_c)

        #print(kmeans.predict(K))
        preds = KMeans(n_clusters=n_classes, random_state=0).fit(U_Phi_c)
        #preds = SpectralClustering(n_clusters=n_classes, affinity='nearest_neighbors').fit(WX)
        ri = adjusted_rand_score(labels_true=targets_train.ravel(), labels_pred=preds.labels_)
        #ri = adjusted_mutual_info_score(labels_true=targets_train, labels_pred=preds)
        print(ri)
        fourier_times.append(time() - start)

        fourier_scores.append(ri)

        feature_map_sparse = SparseSampler(s_plus=s_plus, s_minus=s_minus, gamma=.031, random_state=1, n_components=D)
        WX = feature_map_sparse.fit_transform(data_train)
        K = WX @ np.transpose(WX)
        #K = nearestPD(K)
        #print(isPD(K))
        U_Phi_c,D_Phi_c,_ = np.linalg.svd(K)
        U_Phi_c = U_Phi_c[:,:n_classes+1]
        preds = KMeans(n_clusters=n_classes, random_state=0).fit(U_Phi_c)
        #preds = SpectralClustering(n_clusters=n_classes, affinity='nearest_neighbors').fit(WX)
        ri = adjusted_rand_score(labels_true=targets_train.ravel(), labels_pred=preds.labels_)
        print(ri)
        sparse_scores.append(ri)

    # plot the results:
    pl.figure(figsize=(8, 8))

    pl.plot(samples, fourier_scores)
    pl.figure(figsize=(8, 8))
    pl.plot(samples, sparse_scores)


    pl.show()

if __name__=='__main__':
    main()



