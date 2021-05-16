#!/usr/bin/env python3
"""
Solves the Bayes-optimal state-evolution equations for reconstructing
patterns from the connectivity matrix of a rectified Hopfield model with a
single sample.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 2018/03/20

"""

import argparse
import sys

import numpy as np
import numpy.random as rnd

from scipy import integrate

from matfact import amp  # for some constants, e.g. for initialisation
import hopfield as hf

# maximum simulation time
T_MAX = 20000

DESCRIPTION = '''
This is AMP for pattern reconstruction in the rectified Hopfield model.
'''


def get_mse(prior, a, rho=None):
    """
    Computes the mse for the Hopfield model.

    Parameters
    ----------
    prior :
        numerical code for the prior
    x :
        = a/Delta
    rho :
        if not None, gives the fraction of non-zero elements of the patterns for
        sparse priors. Default: None.
    """
    mse = None

    if prior == hf.PRIOR_HF:
        if rho is None:
            mse = 1 - a
        else:
            mse = rho - a
    elif prior == hf.PRIOR_TSODYKS:
        mse = (1-rho)*rho - a
    else:
        raise NotImplementedError("MSE for the prior %d has not been "
                                  "implemented yet" % prior)

    return mse


def integrand_hf(w, x, rho=None):
    """
    Defines the integrand for the state evolution of the rectified Hopfield
    model with standard Rademacher prior.

    The first argument is the one that we are integrating over.

    Parameters
    ----------
    w :
        Standard Gaussian iid
    x :
        = a/Delta
    """
    return np.exp(-w**2/2)/np.sqrt(2*np.pi) * np.tanh(x + np.sqrt(x)*w)


def integrand_hf_sparse(w, x, rho=None):
    """
    Defines the integrand for the state evolution of the rectified Hopfield
    model with Rademacher-Bernoulli prior, a.k.a. the sparse Hopfield model.

    The first argument is the one that we are integrating over.

    Parameters
    ----------
    w :
        Standard Gaussian iid
    x :
        = a/Delta
    rho :
        the fraction of non-zero elements of the patterns
    """
    return (np.exp(-w**2/2)/np.sqrt(2*np.pi) *
            rho**2 * np.exp(-x/2) * np.sinh(x + np.sqrt(x)*w) /
            (1 + rho * (np.exp(-x/2) * np.cosh(x + np.sqrt(x)*w) - 1)))


def integrand_tsodyks(w, x, rho):
    """
    Defines the integrand for the state evolution of the rectified Hopfield
    model with a prior related to the Tsodyks model (EPL 1988).

    The first argument is the one that we are integrating over.

    Parameters
    ----------
    w :
        Standard Gaussian iid
    x :
        = a/Delta
    rho :
        0 < rho < 1
    """
    pdf = np.exp(-w**2/2)/np.sqrt(2*np.pi)
    return pdf*(np.exp(w*np.sqrt(x))*(-1 + np.exp(x))*(-1 + rho)**2*rho**2) / \
        ((-(np.exp(x/2.)*(-1 + rho)) + np.exp(w*np.sqrt(x))*rho) *
         (1 + (-1 + np.exp(w*np.sqrt(x) + x/2.))*rho))


def se(prior, rho=None, init=amp.INIT_UNINFORMATIVE):
    integrand = None
    if prior == hf.PRIOR_HF:
        if rho is None:
            integrand = integrand_hf
        else:
            integrand = integrand_hf_sparse
    elif prior == hf.PRIOR_TSODYKS:
        integrand = integrand_tsodyks

    Delta_C = hf.get_Delta_c(prior, rho)
    Deltas = np.linspace(0.05*Delta_C, 2.2*Delta_C, 200)

    for Delta in Deltas:
        if init == amp.INIT_UNINFORMATIVE:
            a = 0 + 1e-6*rnd.rand()  # N.B. that a >= 0 at all times.
        else:
            # informative initialisation with perturbation
            if prior == hf.PRIOR_HF:
                if rho is None:
                    a = 1 - 1e-6*rnd.rand()
                else:
                    a = rho - 1e-6*rnd.rand()
            elif prior == hf.PRIOR_TSODYKS:
                a = rho*(1-rho) - 1e-6*rnd.rand()

        for t in range(T_MAX):
            a_next, err = integrate.quad(integrand, -8, 8,
                                         args=(a/Delta, rho,))

            if np.abs(a_next - a) < 1e-12 and t > 20:
                break
            a = a_next
        mse = get_mse(prior, a, rho)
        print('%8g, %8g, %8g, %8g, %8g' % (rho if rho is not None else np.nan,
                                           Delta, Delta / Delta_C, t, mse))
    print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    help_prior = ("Prior: %d=Hopfield (default), %d=Tsodyks" %
                  (hf.PRIOR_HF, hf.PRIOR_TSODYKS))
    parser.add_argument("--prior", type=int, help=help_prior,
                        default=hf.PRIOR_HF)
    parser.add_argument('--rho', type=float, default=None,
                        help='Fraction of non-zero entries in the inputs for '
                        'sparse inputs, Tsodyks parameter.')
    init_help = ("initialisation: %d=uninformative, %d=informative" %
                 (amp.INIT_UNINFORMATIVE, amp.INIT_INFORMATIVE))
    parser.add_argument('-i', '--init', type=int,
                        default=amp.INIT_UNINFORMATIVE, help=init_help)
    args = parser.parse_args()

    rnd.seed(0)

    if args.prior not in hf.PRIORS:
        raise ValueError("prior must be between %d and %d." %
                         (min(hf.PRIORS), max(hf.PRIORS)))
        exit()
    if args.prior == hf.PRIOR_TSODYKS and args.rho is None:
        raise ValueError("Need non-zero rho for Tsodyks-like prior")

    prior_name = hf.prior_name(args.prior, args.rho)

    fname = ("se_%s_P1_rho%g_i%d.dat" %
             (hf.prior_name(args.prior, args.rho, True),
              args.rho, args.init))
    logfile = open(fname, "w")
    sys.stdout = logfile

    print("# SE for the scalar order parameter in the rectified Hopfield model")
    print("# %s prior with %sinformed initialisation \n" % (prior_name,
          ("un" if args.init == amp.INIT_UNINFORMATIVE else "")))
    print('# %8s, %8s, %8s, %8s, %8s' %
          ('rho', 'Delta', 'Delta/Delta_c', 't', 'mse'))
    se(args.prior, args.rho, args.init)
