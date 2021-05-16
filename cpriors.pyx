# cython: profile=True

"""
CPython implementation of threshold functions f_in for representative priors for
approximate message passing for low-rank matrix reconstruction.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180323

"""

cimport cython

from libc.math cimport exp, fabs
from libc.math cimport tanh

import numpy as np

# data type for our arrays
DTYPE = np.float64

ctypedef double (*f_in)(double, double, double)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double f_in_hopfield_scalar(double A, double B, double rho):
    """
    Fixed-point function for the Hopfield prior with x\in{-1, 1}.
    """
    return tanh(B)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double f_in_hopfield_sparse_scalar(double A, double B, double rho):
    """
    Fixed-point function for the sparse Hopfield prior with x\in{-1, 0, 1}.
    """
    return ((-1 + exp(2*B))*rho)/(-2*exp(A/2. + B)*(rho - 1) + rho + exp(2*B)*rho)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef double f_in_tsodyks_scalar(double A, double B, double rho):
    """
    Fixed-point function for the Tsodyks-like prior with x\in{-rho, 1-rho}.
    """
    return 1 - rho - (1 - rho) / (1 - rho + exp(B + A*(rho - 0.5))*rho)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef f_in_hopfield_mf_mean(double [:, :] A, double [:, :] B, double rho):
    return mf_mean(A, B, rho, &f_in_hopfield_scalar)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef f_in_hopfield_sparse_mf_mean(double [:, :] A, double [:, :] B, double rho):
    return mf_mean(A, B, rho, &f_in_hopfield_sparse_scalar)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef f_in_tsodyks_mf_mean(double [:, :] A, double [:, :] B, double rho):
    return mf_mean(A, B, rho, &f_in_tsodyks_scalar)


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef mf_mean(double [:, :] A, double [:, :] B, double rho, f_in f_in_fun):
    """
    Computes the mean of the given threshold function using a mean-field
    approximation.

    Parameters:
    -----------
    A : (r, r)
        one covariance matrix
    B : (n, r)
        N means
    b0 : scalar
        0 if this is not a sparse hopfield model, else .5 * np.log(rho / (1 -
        rho)), where rho is the element-wise probability that a an element of x
        is zero

    Returns:
    --------
    x : (n, r)
        N updated estimators
    var : (r, r)
        updated average covariance matrix

    """
    cdef size_t i, t, k, k2
    cdef size_t n = B.shape[0]
    cdef size_t r = B.shape[1]
    cdef double eps_conv = 1e-7  # absolute mean difference per spin component
    cdef size_t max_iter = 50  # max number of mean field iterations

    # Iterate mean-field sequentially for convergence reasons,
    # hence the loops are unavoidable
    # See e.g. David Barber, Bayesian Reasoning and Machine Learning, Sec. 28.4
    m_np = np.zeros((n, r), dtype=DTYPE)
    cdef double [:, :] m = m_np
    m_old_np = np.zeros((r), dtype=DTYPE)
    cdef double [:] m_old = m_old_np
    cdef double Ak_dot_mi  # vector product inside for loop
    cdef double diff

    # avoid the overhead of a function call, just define these two "parameters"
    cdef double arg_B

    # for each estimator, ...
    for i in range(n):
        # ... iterate the mean-field eqn...
        for t in range(max_iter):
            m_old[:] = m[i, :]
            # ... sequentially (!) for k=1,...,R
            for k in range(r):
                Ak_dot_mi = 0  # A[k, :] . m[i, :]
                for k2 in range(r):
                    Ak_dot_mi += A[k, k2] * m[i, k2]
                arg_A = A[k, k]
                arg_B = B[i, k] - Ak_dot_mi + A[k, k] * m[i, k]
                m[i, k] = f_in_fun(arg_A, arg_B, rho)
            diff = 0
            for k in range(r):
                diff += fabs(m[i, k] - m_old[k])
            if (diff / r) < eps_conv:
                break

    return m_np
