"""
Helper methods to study inference in the rectified Hopfield model.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180310
"""

import numpy as np
import numpy.random as rnd

from scipy.special import erfc


PRIOR_HF = 1
PRIOR_TSODYKS = 2
PRIOR_GB = 3  # Gauss-Bernoulli
PRIORS = [PRIOR_HF, PRIOR_TSODYKS, PRIOR_GB]


def get_J_unrectified(xis, nu):
    '''
    Returns the connectivity for the normal Hopfield model with noise *without*
    passing it through a ReLU.

    Parameters:
    -----------
    xis : (N, P)
        P samples in N dimensions
    nu :
        std.dev. of the Gaussian noise
    '''
    n = xis.shape[0]
    noise = nu * rnd.randn(n, n)
    noise = np.triu(noise, k=1) + np.triu(noise).T

    J = 1 / np.sqrt(n) * xis @ xis.T + noise

    # make J go through a ReLU channel if tau is not none
    return J


def get_J(xis, nu, tau=None):
    '''
    Returns the connectivity for the rectified Hopfield model.

    If no explicit threshold tau is given, this function chooses tau s.t. only
    10% of the possible connections are made.

    Parameters:
    -----------
    xis : (N, P)
        P samples in N dimensions
    nu :
        std.dev. of the Gaussian noise.
    tau :
        threshold parameter for the rectified Hopfield model. By default it is
        None, in which case the threshold is chosen such that 10% of upper
        triangular entries of J are non-zero.

    Returns:
    --------
    J : (N, N)
        connectivity matrix of the rectified Hopfield model.
    tau : scalar
        if the given tau is None, returns the tau that was dynamically chosen.
    '''
    N = xis.shape[0]
    J = get_J_unrectified(xis, nu)

    # calculate tau if tau is not given:
    if tau is None:
        # if we want 10% of all connections to be active, have to have 20%
        # non-zero elements in J because of the symmetry of J
        tau = np.sort(J.reshape(N**2))[round(.8*N**2)]
        return np.maximum(0, J-tau), tau
    else:
        return np.maximum(0, J-tau)


def get_S(J, nu, tau):
    '''
    Returns the Fisher score matrix for the given connectivity matrix J.

    Parameters:
    -----------
    J : (N, N)
        observed connectivity matrix
    nu :
        std.dev. of the Gaussian noise
    tau :
        threshold for the ReLU
    '''
    # S(J = 0) =
    SJ0 = -(np.sqrt(2 / np.pi) * np.exp(-0.5 * tau**2 / nu**2) /
            (nu * erfc(-tau / np.sqrt(2) / nu)))
    SJgt0 = (J + tau) / nu**2
    return np.where(J < 1e-9, SJ0, SJgt0)


def binarise(x):
    """
    Returns a thresholded version of the given array.

    Returns an array Y with Y[i, j] = 1 if x[i, j] > 0 and -1 otherwise. Thus in
    particular 0 is converted to -1. It has the same type as the given array.

    """
    return 2 * np.array(x > 0).astype(x.dtype) - 1


def get_samples(N, P, prior=PRIOR_HF, rho=None):
    """
    Creates P patterns in N dimensions with the given prior distribution.
    Some priors have a sparse version, where rho is the fraction of
    non-zero elements.

    Parameters
    ----------
    N :
        dimension of each pattern xi
    P :
        number of patterns
    prior :
        code for the prior distribution of samples.
        Default: PRIOR_HF (Hopfield or Rademacher prior)
    rho :
        if not None, gives the fraction of non-zero elements of the patterns for
        sparse priors. Default: None.
    """
    xis = None
    if prior == PRIOR_HF:
        xis = rnd.choice([-1, 1], (N, P))
        if rho is not None:  # sparse
            support = rnd.binomial(1, rho, (N, P))
            xis = support * xis
    elif prior == PRIOR_TSODYKS:
        binomials = rnd.binomial(1, rho, (N, P))  # 1 trial, p(x=1)=rho, N x P array
        xis = binomials - rho
    elif prior == PRIOR_GB:
        support = rnd.binomial(1, rho, (N, P))
        xis = support * rnd.randn(N, P)
    else:
        raise ValueError("prior must be one of %g (Rademacher), %d (Tsodyks-"
                         "-like), or %d (Gauss-Bernoulli)" % tuple(PRIORS))

    return xis


def get_Delta(nu, tau):
    """
    Results the analytical result for the Fisher information of the rectified
    Hopfield model.

    Parameters:
    -----------
    nu : scalar
        std.dev. of the Gaussian noise
    tau : scalar
        threshold of the ReLU

    Returns:
    --------
    Delta : scalar
        The Fisher information of the output channel used in the rectified
        Hopfield model at :math:`H_{ij} \equiv 1 / \sqrt{N} x_i^\top x_j = 0`.
    """
    t1 = tau * np.exp(-0.5*tau**2/nu**2) / (np.sqrt(2 * np.pi) * nu**3)
    t2 = np.exp(-tau**2/nu**2) / (np.pi * nu**2 * erfc(-tau/np.sqrt(2)/nu))
    t3 = erfc(tau/np.sqrt(2)/nu) / (2*nu**2)

    return 1 / (t1 + t2 + t3)


def get_Delta_c(prior, rho=None):
    """
    Returns the critical value of the effective noise at which the trivial fixed
    point of the state evolution changes stability.

    Parameters:
    -----------
    prior :
        numerical code for the prior
    rho :
        if not None, gives the fraction of non-zero elements of the patterns for
        sparse priors. Default: None.
    """
    if prior == PRIOR_HF:
        Delta_c = 1 if rho is None else rho**2  # (Sparse) Hopfield
    elif prior == PRIOR_TSODYKS:
        Delta_c = rho**2 * (rho-1)**2  # Tsodyks
    else:
        raise ValueError("only implemented for Hopfield and Tsodyk's prior")
    return Delta_c


def prior_name(prior, rho=None, short=False, mf=False):
    """
    Returns a printable description of the name of the prior used.

    Parameters:
    -----------
    prior :
        numerical code for the prior
    rho :
        if not None and prior is Hopfield, will assume this is a parse model.
    short :
        if True, return a short name of the prior suitable for filenames.
    mf :
        whether we are using the meanfield-approx to compute the prior
        (only for AMP program)
    """
    prior_name = None
    if prior == PRIOR_HF:
        if rho is None:
            prior_name = 'hf' if short else 'Hopfield'
        else:
            prior_name = 'hf-sparse' if short else 'sparse Hopfield'
    else:
        prior_name = 'tsodyks' if short else 'Tsodyks'
    if mf and not short:  # mean-field approximation of the prior is employed
        prior_name = 'mean-field ' + prior_name
    return prior_name
