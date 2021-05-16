"""
Implementation of threshold functions f_in for representative priors for
approximate message passing for low-rank matrix reconstruction.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180323

"""

import cpriors

import functools
import itertools
import numpy as np
import numpy.linalg as la


def f_in_gauss(A, B, var_noise=0.5):
    """
    Implements the threshold function for the Gaussian channel with the given
    variance.

    Parameters:
    -----------
    A : (r, r)
    B : (N, r)

    Returns:
    --------
    x : (N, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix

    """
    var = la.inv(1 / var_noise * np.eye(*A.shape) + A)
    x = B @ var

    return x, var


def f_in_jointly_gauss_bernoulli(A, B, rho):
    """
    Threshold function for the prior where every spin is i.i.d. and either zero
    at once or drawn from a multinormal distribution with mean 0 and covariance
    being the identity.

    Parameters:
    -----------
    rho :
        fraction of spin elements that are non-zero

    Returns:
    --------
    x : (N, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix
    """
    inverse = la.inv(A + np.eye(*A.shape))  # (r, r)
    root = np.sqrt(la.det(A + np.eye(*A.shape)))  # scalar
    expo = np.diag(np.exp(-0.5 * B @ inverse @ B.T))  # (N,)
    # TODO: make me more effective...
    # matrix = 1 / N * inverse @ B.T @ B @ inverse
    matrix = (B @ inverse.T)[:, None] * (B @ inverse.T)[:, :, None]  # (N, r, r)
    # assert(np.allclose(matrix[N-1],
    #                    inverse @ np.outer(B[N-1], B[N-1]) @ inverse))
    # assert(np.allclose(np.mean(matrix, axis=0),
    #                    1 / N * inverse @ B.T @ B @ inverse))

    mean = rho * B @ inverse.T / ((1-rho) * root * expo[:, None] + rho)
    var = rho * inverse / ((1 - rho) * root * expo[:, None, None] + rho)

    var += (rho * (1-rho) * root * expo[:, None, None] /
            ((1 - rho) * root * expo[:, None, None] + rho)**2) * matrix

    return mean, np.mean(var, axis=0)


def f_in_gauss_bernoulli_assuming_diagA(A, B, rho):
    """
    Threshold function for the prior where every element of every spin is i.i.d.
    drawn from a Gauss Bernoulli distribution, ASSUMING THAT A IS DIAGONAL!

    Parameters:
    -----------
    rho :
        fraction of spin elements that are non-zero

    Returns:
    --------
    x : (N, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix
    """
    B2 = B**2
    expB2over2 = np.exp(B2/2)
    x = B + rho * B / (expB2over2 * (rho - 1) - rho)
    # the variance matrix is diagonal!
    var = (np.exp(B2) * (rho - 1)**2 + (1 + B2) * expB2over2 *
           (1 - rho) * rho) / (expB2over2 * (1 - rho) + rho)**2  # (N, r)

    return x, np.diag(np.mean(var, axis=0))


def f_in_hopfield(A, B, rho=None, diagonalA=False):
    """
    Implements the threshold function for the Hopfield model, where each
    component of every spin is +/- 1 with probability 1/2.

    rho is the optional sparsity of the data matrix. If rho is not None, but a
    number between 0 and 1, it is the probability that any component of any spin
    is non-zero.

    Parameters:
    -----------
    A : (r, r)
    B : (n, r)  
    rho : scalar
        element-wise probability that a an element of x is non-zero
    diagonalA :
        if True, assume that the covariance matrix A is diagonal to speed up
        the computation. Default is False.

    Returns:
    --------
    x : (n, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix

    """
    n, r = B.shape

    # These are calls to hashed functions
    xis = _hopfield_get_all_rows(r, rho)  # (3^r, r)
    pxis = _hopfield_get_all_rows_probabilities(r, rho)  # (3^r)

    if diagonalA:
        exponents = -0.5 * xis**2 @ np.diag(A) + B @ xis.T  # (N, 3^r)
    else:
        #       this |  np.diag does not mean that we assume that A is diagonal!
        #           \ /  c.f. the relevant unit tests for detailed information.
        exponents = np.diag(-0.5 * xis @ A @ xis.T) + B @ xis.T  # (N, 3^r)
    exponents_max = np.max(exponents, axis=1)  # for numerical stability
    pxis_exps = np.exp(exponents - exponents_max[:, None]) * pxis  # (N, 3^r)
    Z = np.sum(pxis_exps, axis=1)  # N
    ps = 1 / Z[:, None] * pxis_exps  # (N, 3^r)

    x = ps @ xis  # means, (N, P)
    xisXisT = xis[:, None] * xis[:, :, None]  # outer prods, (3^r, P, P)
    var = np.mean(np.tensordot(ps, xisXisT, axes=1), axis=0) - x.T @ x / n

    logZ = np.log(np.sum(pxis * np.exp(exponents), axis=-1))

    return x, var, logZ


def f_in_hopfield_mf(A, B, rho=None):
    """
    Implements the threshold function for the Hopfield model using a mean-field
    approximation.

    Parameters:
    -----------
    A : (r, r)
    B : (n, r)
    rho : scalar
        element-wise probability that a an element of x is non-zero

    Returns:
    --------
    x : (n, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix
    """
    def f_cov(A, B, rho):
        if rho is None or rho == 1:
            return 1 - np.tanh(B) ** 2
        else:
            return (2*np.exp(B)*rho*(-(np.exp(A/2.)*(1 + np.exp(2*B))*(rho - 1)) + 2*np.exp(B)*rho)) / \
                (-2*np.exp(A/2. + B)*(rho -1) + rho + np.exp(2*B)*rho)**2

    def f_DKL(A, B, rho):
        if rho is None or rho == 1:
            return -(np.log(2/(1 + np.exp(2*B)))/(1 + np.exp(2*B))) \
                - (np.log(1 + np.tanh(B))*(1 + np.tanh(B)))/2.
        else:
            ZW = -2*np.exp(A/2. + B)*(-1 + rho) + rho + np.exp(2*B)*rho
            return -((rho*np.log(2/ZW))/ZW) - (np.exp(2*B)*rho*np.log((2*np.exp(2*B))/ZW))/ZW + \
                (2*np.exp(A/2. + B)*(-1 + rho)*np.log((-2*np.exp(A/2. + B)*(-1 + rho))/((1 - rho)*ZW)))/ZW

    if rho is None:
        # rho = None means non-sparse prior, which corresponds to rho=1;
        # the value of rho is not used by the function f_in_hopfield_mf_mean
        # anyway
        f_mean = cpriors.f_in_hopfield_mf_mean
    else:
        f_mean = cpriors.f_in_hopfield_sparse_mf_mean

    rho = 1 if rho is None else rho

    return f_in_mf(A, B, f_mean, f_cov, f_DKL, rho=rho)


def f_in_tsodyks(A, B, rho):
    """
    Implements the threshold function for a model inspired by the Tsodyks
    model [1, 2].

    rho is a parameter between 0 and 1.

    [1] M. V Tsodyks and M. V Feigel’man, Europhys. Lett. 6, 101 (1988).
    [2] M. V. Tsodyks, Europhys. Lett. 7, 203 (1988).

    Parameters:
    -----------
    A : (r, r)
    B : (n, r)
    rho : scalar
        0 < rho < 1

    Returns:
    --------
    x : (n, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix
    """
    n, r = B.shape

    # These are calls to hashed functions
    xis = _tsodyks_get_all_rows(r, rho)  # (3^r, r)
    pxis = _tsodyks_get_all_rows_probabilities(r, rho)  # (2^r)

    #       this |  np.diag does not mean that we assume that A is diagonal!
    #           \ /  c.f. the relevant unit tests for detailed information.
    exponents = np.diag(-0.5 * xis @ A @ xis.T) + B @ xis.T  # (N, 2^r)
    exponents_max = np.max(exponents, axis=1)  # for numerical stability
    pxis_exps = np.exp(exponents - exponents_max[:, None]) * pxis  # (N, 2^r)
    Zs = np.sum(pxis_exps, axis=1)  # N
    ps = 1 / Zs[:, None] * pxis_exps  # (N, 3^r)

    x = ps @ xis  # means, (N, P)
    xisXisT = xis[:, None] * xis[:, :, None]  # outer prods, (2^r, r, r)
    var = np.mean(np.tensordot(ps, xisXisT, axes=1), axis=0) - x.T @ x / n

    logZ = np.log(np.sum(pxis * np.exp(exponents), axis=-1))

    return x, var, logZ


def pX_tsodyks(x, rho):
    """
    Prior distribution for vectors in the Tsodyks-like model.

    Parameters:
    -----------
    x : (N)
        input pattern
    rho :
        0 < rho < 1
    """
    p = rho * np.ones(x.shape)  # pr(x_i = 1-rho)
    p[x < 0] = 1 - rho  # pr(x_i = -rho)

    return p.prod(axis=-1)


def f_in_tsodyks_mf(A, B, rho=None):
    """
    Implements the threshold function for the Hopfield model using a mean-field
    approximation.

    Parameters:
    -----------
    A : (r, r)
    B : (n, r)
    rho : scalar
        element-wise probability that a an element of x is non-zero

    Returns:
    --------
    x : (n, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix
    """
    def f_cov(A, B, rho):
        return (np.exp(B + A*(-0.5 + rho))*(1 - rho)*rho) / \
            (1 - rho + np.exp(B + A*(-0.5 + rho))*rho)**2

    def f_DKL(A, B, rho):
        ZW1 = np.exp(A/2. - B - A*rho)
        ZW2 = np.exp(B + A*(-0.5 + rho))
        return -(np.log(1/(-(ZW1*(-1 + rho)) + rho))/(1 + ZW1*(-1 + 1/rho))) + \
            ((-1 + rho)*np.log(1/(1 + (-1 + ZW2)*rho)))/(1 + (-1 + ZW2)*rho)

    f_mean = cpriors.f_in_tsodyks_mf_mean

    return f_in_mf(A, B, f_mean, f_cov, f_DKL, rho=rho)


def f_in_mf(A, B, f_mean, f_cov, f_DKL=None, **kwargs):
    """
    Implements the threshold function using a mean-field approximation.

    Parameters:
    -----------
    A : (r, r)
    B : (n, r)
    f_mean, f_cov:
        function to compute the mean, covariance and entropy of the single-spin
        probability distribution.
    kwargs :
        additional parameters for use by the functions computing the mean, etc.

    Returns:
    --------
    x : (n, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix
    logZ : (n)
        variational lower bound on the logarithm of the partition function of
        the posterior density of each spin.
    """
    N, r = B.shape

    m = f_mean(A, B, **kwargs)

    diag_A = np.diag(np.diag(A))
    # parameters of the mean-field approximation of the prior, used below
    tildeA = np.tile(np.diag(A), (N, 1))
    tildeB = B - m @ A.T + m @ diag_A.T

    # Compute variances
    covs = f_cov(tildeA, tildeB, **kwargs)  # (n, r)
    # and assign non-diagonal elements via a linear response approximation
    cov_avg = 1 / N * (np.diag(np.sum(covs, 0))
                       + (1. - np.eye(r)) * (-.5 * A * (covs.T @ covs)))

    # obtain a variational lower bound on log Z for the N distributions
    if f_DKL is not None:
        logZ = np.sum(f_DKL(tildeA, tildeB, **kwargs), -1) +  \
               - 0.5 * np.sum((m @ A) * m, -1) + np.sum(B * m, -1)
    else:
        logZ = np.nan

    return m, cov_avg, logZ


def pX_hopfield_sparse(x, rho):
    """
    Prior distribution for vectors in the sparse Hopfield model.

    Parameters:
    -----------
    x : (N)
        input pattern
    rho :
        probability that a component of the input pattern is non-zero
    """
    p = 0.5 * rho * np.ones(x.shape)  # pr(x_i = +-1)
    p[x == 0] = 1 - rho  # pr(x_i = 0)

    return p.prod(axis=-1)


def f_in_rademacher_bernoulli(A, B, rho):
    """
    Threshold function for the Rademacher-Bernoulli prior, which is equivalent
    to the sparse Hopfield model.

    Parameters:

    -----------
    rho :
        probability that a component of the input pattern is non-zero
    """
    return f_in_hopfield(A, B, rho)


def f_in_sbm(A, B, avgCov=True):
    """
    Implements the threshold function for the stochastic block model.

    This threshold function also applies to the degree-corrected SBM and can
    handle these cases:
        - The matrix A has shape (r, r) and is the same for all N distributions
        - The matrix A has shape (N, r, r), i.e. one (r, r) matrix for each
          distribution.

    Parameters:
    -----------
    A : (N, r, r)
    B : (N, r)
    avgCov : bool
        if True, return the averaged covariance matrix. Otherwise, return N
        covariance matrices, one for every node.

    Returns:
    --------
    x : (N, r)
        N updated estimators
    var : (N, r, r) or (r, r)
        N covariance matrices or the average covariance matrix, depending on the
        parameter ``avgCov``.
    """
    N, r = B.shape

    # normalisation constant for all N distributions
    args = B - np.diagonal(A, axis1=-2, axis2=-1) / 2  # (N, r)
    max_args = np.max(args, axis=-1)  # for numerical stability
    exps = np.exp(args - max_args[:, None])
    Z = np.sum(exps, axis=-1)

    x = 1 / Z[:, None] * exps  # means, f_in

    if avgCov:
        var = 1 / N * (np.diag(np.sum(x, axis=0)) - x.T @ x)  # covariance
    else:
        var = (x[:, None] * np.eye(r)  # N (r,r) matrices with x on the diagonal
               - x[:, None] * x[:, :, None])

    # To compute the correct partition functions, have to remove the max_arg
    # term we introduced above for numerical stability
    logZ = np.log(np.sum(1 / r * np.exp(args), axis=-1))

    return x, var, logZ


@functools.lru_cache(maxsize=2)
def _hopfield_get_all_rows_probabilities(r, rho=None):
    """
    Returns the probability distribution over all the possible rows that can
    appear when stacking r Hopfield patterns in a matrix X, with each pattern
    stored in one column of X.

    The rows are in the order given by the method _get_all_hopfield_rows.

    Parameters:
    -----------
    rho :
        probability that a component of the input pattern is non-zero
    """
    if rho is None:  # dense
        return 1 / (2**r) * np.ones((2**r))

    # sparse
    return pX_hopfield_sparse(_hopfield_get_all_rows(r, True), rho)


@functools.lru_cache(maxsize=2)
def _hopfield_get_all_rows(r, rho=None):
    """
    Returns all the possible rows that can appear when stacking r Hopfield
    patterns in a matrix X, with each pattern stored in one column of X.

    rho :
        probability that a component of the input pattern is non-zero
    """
    if rho is None:  # dense
        xis = np.array([x for x in itertools.product([-1, 1], repeat=r)])
    else:  # sparse
        xis = np.array([x for x in itertools.product([-1, 0, 1], repeat=r)])

    return xis   # (2^r, r) or (3^r, r), respectively


@functools.lru_cache(maxsize=2)
def _tsodyks_get_all_rows_probabilities(r, rho):
    """
    Returns the probability distribution over all the possible rows that can
    appear when stacking r Tsodyks-like patterns in a matrix X, with each
    pattern stored in one column of X.

    The rows are in the order given by the method _get_all_tsodyks_rows.

    Parameters:
    -----------
    rho :
        probability that a component of the input pattern is non-zero
    """
    return pX_tsodyks(_tsodyks_get_all_rows(r, rho), rho)


@functools.lru_cache(maxsize=2)
def _tsodyks_get_all_rows(r, rho):
    """
    Returns all the possible rows that can appear when stacking r Tsodyks-like
    patterns in a matrix X, with each pattern stored in one column of X.

    rho :
        probability that a component of the input pattern is non-zero
    """
    # (2^r, r)
    return np.array([x for x in itertools.product([-rho, 1-rho], repeat=r)])
