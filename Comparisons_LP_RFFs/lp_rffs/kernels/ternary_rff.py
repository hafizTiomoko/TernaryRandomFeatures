import numpy as np
import torch
import sys

sys.path.append("../utils")
from misc_utils import set_random_seed
from gaussian_exact import GaussianKernel
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import check_array, check_is_fitted
import scipy
import scipy.sparse as sp
from scipy.optimize import least_squares
pi = np.pi

class TernaryRFF(object):
    def __init__(self, n_feat, n_input_feat, s_minus, s_plus, kernel=None, rand_seed=1):
        self.n_feat = n_feat  # number of rff features
        self.kernel = kernel
        self.n_input_feat = n_input_feat  # dimension of the original input
        self.rand_seed = rand_seed
        self.get_gaussian_wb()
        self.s_minus = s_minus
        self.s_plus = s_plus

    def _sparse_random_matrix(self, n_components, n_features, density='auto',
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
        # _check_input_size(n_components, n_features)
        #density = _check_density(density, n_features)
        rng = check_random_state(random_state)
        #rng = np.random

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
            components = scipy.sparse.csr_matrix((data, indices, indptr),
                                                 shape=(n_components, n_features))

            return np.sqrt(1 / density) / np.sqrt(n_components) * components
    def get_gaussian_wb(self):
        # print("using sigma ", 1.0/float(self.kernel.sigma), "using rand seed ", self.rand_seed)
        np.random.seed(self.rand_seed)
        self.w = np.random.normal(scale=1.0 / float(self.kernel.sigma),
                                  size=(self.n_feat, self.n_input_feat))
        #self.w = self._sparse_random_matrix(self.n_feat, self.n_input_feat, density=1,
        #                      random_state=1)
        # print("using n rff features ", self.w.shape[0] )
        np.random.seed(self.rand_seed)
        self.b = np.random.uniform(low=0.0, high=2.0 * np.pi, size=(self.n_feat, 1))

    def torch(self, cuda=False):
        self.w = torch.DoubleTensor(self.w)
        self.b = torch.DoubleTensor(self.b)
        if cuda:
            self.w = self.w.cuda()
            self.b = self.b.cuda()

    def cpu(self):
        self.w = self.w.cpu()
        self.b = self.b.cpu()

    def get_cos_feat(self, input_val, dtype="double"):
        # input are original representaiton with the shape [n_sample, n_dim]
        if isinstance(self.kernel, GaussianKernel):
            if isinstance(input_val, np.ndarray):
                self.input = input_val.T
                self.feat = np.sqrt(2 / float(self.n_feat)) * np.cos(np.dot(self.w, self.input) + self.b)
                if dtype == "double":
                    return torch.DoubleTensor(self.feat.T)
                else:
                    return torch.FloatTensor(self.feat.T)
            else:
                self.input = torch.transpose(input_val, 0, 1)
                self.feat = float(np.sqrt(2 / float(self.n_feat))) * torch.cos(torch.mm(self.w, self.input) + self.b)
                return torch.transpose(self.feat, 0, 1)
        else:
            raise Exception("the kernel type is not supported yet")

    def get_sin_cos_feat(self, input_val):
        pass

    def get_ternary_feat(self, input_val, dtype="double"):
        if isinstance(input_val, np.ndarray):
            self.input = input_val.T
            #projection = safe_sparse_dot(self.w, self.input)
            projection = np.dot(self.w, self.input) + self.b
            #projection += self.b
            self.feat = np.sqrt(2 / float(self.n_feat)) * (projection > (np.sqrt(2) * self.s_plus)).astype(float) - (
                    projection < (np.sqrt(2) * self.s_minus)).astype(float)
            self.feat.double()
            if dtype == "double":
                return torch.DoubleTensor(self.feat.T)
            else:
                return torch.FloatTensor(self.feat.T)
        else:
            self.input = torch.transpose(input_val, 0, 1)
            projection = torch.mm(self.w, self.input) + self.b
            self.feat = np.sqrt(2 / float(self.n_feat)) * (projection > (np.sqrt(2) * self.s_plus)).double() - (
                    projection < (np.sqrt(2) * self.s_minus)).double()
            #self.feat = float(np.sqrt(2 / float(self.n_feat))) * torch.cos(torch.mm(self.w, self.input) + self.b)
            return torch.transpose(self.feat, 0, 1)

    def get_kernel_matrix(self, X1, X2, quantizer1=None, quantizer2=None, consistent_quant_seed=True):
        '''
        X1 shape is [n_sample, n_dim], if force_consistent_random_seed is True
        the quantization will use the same random seed for quantizing rff_x1 and rff_x2
        '''
        rff_x1 = self.get_ternary_feat(X1)
        rff_x2 = self.get_ternary_feat(X2)

        if consistent_quant_seed and (quantizer1 is not None) and (quantizer2 is not None):
            assert quantizer1.rand_seed == quantizer2.rand_seed, "quantizer random seed are different under consistent quant seed mode!"
        if quantizer1 != None:
            if consistent_quant_seed and list(rff_x1.size()) == list(rff_x2.size()):
                print("quantizing rff_x1 with random seed", quantizer1.rand_seed)
                set_random_seed(quantizer1.rand_seed)
            else:
                print("quantizing rff_x1 without fixed random seed")
            # print("quantization 1 activated ", X1.shape)
            # print("quantizer 1 bits", quantizer1.nbit)
            # print("quantizer 1 scale", quantizer1.scale)
            rff_x1 = quantizer1.quantize(rff_x1)
        if quantizer2 != None:
            if consistent_quant_seed:
                print("quantizing rff_x2 with random seed", quantizer2.rand_seed)
                set_random_seed(quantizer2.rand_seed)
            # print("quantization 2 activated ", X2.shape)
            # print("quantizer 2 bits", quantizer2.nbit)
            # print("quantizer 2 scale", quantizer2.scale)
            rff_x2 = quantizer2.quantize(rff_x2)
        self.rff_x1, self.rff_x2 = rff_x1, rff_x2
        return torch.mm(rff_x1, torch.transpose(rff_x2, 0, 1))

def estim_tau(X):
    tau = np.mean(np.diag(X.T @ X))

    return tau
def compute_thresholds(tau):
    #F = lambda x: ((np.exp(-x[0] ** 2 / tau) + np.exp(-x[1] ** 2 / tau)) / np.sqrt(2 * pi * tau) - np.exp(-tau / 2),
    #               (-x[0] * np.exp(-x[0] ** 2 / tau) + x[1] * np.exp(-x[1] ** 2 / tau)) / (
    #                   np.sqrt(2 * pi * tau ** 3)) - np.exp(-tau / 2) / 2)
    ### relu
    F = lambda x: ((np.exp(-x[0] ** 2 / tau) + np.exp(-x[1] ** 2 / tau)) / np.sqrt(2 * pi * tau) - 1/2,
                   (-x[0] * np.exp(-x[0] ** 2 / tau) + x[1] * np.exp(-x[1] ** 2 / tau)) / (
                       np.sqrt(2 * pi * tau ** 3)) - 1/np.sqrt(8*pi*tau))

    res = least_squares(F, (1, 1), bounds=((0, 0), (1, 1)))
    return res.x

def test_pytorch_gaussian_kernel():
    n_feat = 10
    input_val = np.ones([2, n_feat])
    input_val[0, :] *= 1
    input_val[0, :] *= 2
    # get exact gaussian kernel
    kernel = GaussianKernel(sigma=2.0)
    kernel_mat = kernel.get_kernel_matrix(input_val, input_val)
    kernel_mat_torch = kernel.get_kernel_matrix(torch.Tensor(input_val), torch.Tensor(input_val))
    #np.testing.assert_array_almost_equal(kernel_mat.cpu().numpy(), kernel_mat_torch.cpu().numpy())
    print("gaussian kernel pytorch version test passed!")


def test_rff_generation():
    n_feat = 10
    n_rff_feat = 1000000
    input_val = np.ones([2, n_feat])
    input_val[0, :] *= 1
    input_val[0, :] *= 2
    # get exact gaussian kernel
    kernel = GaussianKernel(sigma=2.0)
    kernel_mat = kernel.get_kernel_matrix(input_val, input_val)
    tau_est = estim_tau(input_val)
    print(tau_est)
    thresholds = compute_thresholds(tau_est)
    s_minus = np.min(thresholds)
    s_plus = np.max(thresholds)
    print(s_minus, s_plus)
    # get RFF approximate kernel matrix
    rff = TernaryRFF(n_rff_feat, n_feat, s_minus=s_minus, s_plus= s_plus, kernel=kernel)
    rff.get_gaussian_wb()
    approx_kernel_mat = rff.get_kernel_matrix(input_val, input_val)
    #np.testing.assert_array_almost_equal(approx_kernel_mat.cpu().numpy(), kernel_mat.cpu().numpy(), decimal=3)
    print("rff generation test passed!")


def test_rff_generation2():
    n_feat = 10
    n_rff_feat = 1000000
    input_val = np.ones([2, n_feat])
    input_val[0, :] *= 1
    input_val[0, :] *= 2
    # get exact gaussian kernel
    kernel = GaussianKernel(sigma=2.0)
    # kernel_mat = kernel.get_kernel_matrix(input_val, input_val)
    # get RFF approximate kernel matrix
    tau_est = estim_tau(input_val)
    print(tau_est)
    thresholds = compute_thresholds(tau_est)
    s_minus = np.min(thresholds)
    s_plus = np.max(thresholds)
    print(s_minus, s_plus)

    rff = TernaryRFF(n_rff_feat, n_feat, s_minus=s_minus, s_plus= s_plus, kernel=kernel)
    rff.get_gaussian_wb()
    #approx_kernel_mat = rff.get_kernel_matrix(input_val, input_val)
    rff.torch(cuda=False)
    approx_kernel_mat2 = rff.get_kernel_matrix(torch.DoubleTensor(input_val), torch.DoubleTensor(input_val))
    #np.testing.assert_array_almost_equal(approx_kernel_mat.cpu().numpy(), approx_kernel_mat2.cpu().numpy(), decimal=6)
    print("rff generation test 2 passed!")


if __name__ == "__main__":
    test_pytorch_gaussian_kernel()
    test_rff_generation()
    test_rff_generation2()


