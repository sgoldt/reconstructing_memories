#!/usr/bin/env python

"""
Approximate message passing for pattern reconstruction in the rectified
Hopfield model.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180319

"""

import argparse

import numpy.random as rnd

import amp
import priors
import hopfield as hf

DESCRIPTION = '''
This is AMP for pattern reconstruction in the rectified Hopfield model.
'''

N_DEFAULT = 1000
T_MAX = 100


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-N', '--N', type=int, default=N_DEFAULT,
                        help='number of nodes')
    parser.add_argument('-P', '--P', type=int, default=2,
                        help='number of patterns')
    parser.add_argument('--num_runs', type=int, default=1, help="number of"
                        " independent runs of of AMP with the same data")
    parser.add_argument('--nu', type=float, default=.1,
                        help="variance of Gaussian noise")
    parser.add_argument('-t', '--tau', type=float, default=None,
                        help='ReLU threshold; if not specified, it is chosen '
                             's.t. that connectivity is 10%%')
    prior_help = ('prior: %d=Hopfield (incl. sparse), %d=Tsodyks-like' %
                  (hf.PRIOR_HF, hf.PRIOR_TSODYKS))
    parser.add_argument('--prior', type=int, default=hf.PRIOR_HF,
                        help=prior_help)
    parser.add_argument('--rho', type=float, default=None,
                        help='Fraction of non-zero entries in the inputs for '
                        'sparse inputs, Tsodyks parameter.')
    parser.add_argument('--no-damping', action='store_true',
                        help='turn off adaptive damping')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random number generator seed')
    init_help = ('initialisation: %d=uninformative, %d=informative, '
                 '%d=spectral' %
                 (amp.INIT_UNINFORMATIVE, amp.INIT_INFORMATIVE,
                  amp.INIT_SPECTRAL))
    parser.add_argument('-i', '--init', type=int,
                        default=amp.INIT_UNINFORMATIVE, help=init_help)
    parser.add_argument('--mf', action='store_true',
                        help='use the mean-field approximation of the prior')
    args = parser.parse_args()

    if args.rho == 1:
        args.rho = None

    (N, P, nu, tau, rho, init, num_runs) = (args.N, args.P, args.nu, args.tau,
                                            args.rho, args.init, args.num_runs)

    # check that the initialisation parameter is valid:
    if init not in amp.INITIALISATIONS:
        raise ValueError("init must be between 1 and 6. See matfact help!\n"
                         "Will exit now...")
        exit()
    if rho is not None and (rho < 0 or rho > 1):
        raise ValueError("Sparsity must be a real number between 0, 1."
                         "See the help! Will exit now...")
        exit()
    if args.prior == hf.PRIOR_TSODYKS and rho is None:
        raise ValueError("need to specifiy rho for Tsodyk's prior")
    # if args.mf:
        # When using the mean-field approximation, the variational estimates
        # of the partition function are not precise enough to enable damping
        # args.no_damping = True

    # find the correct prior
    prior = None
    if args.prior == hf.PRIOR_HF:
        prior = priors.f_in_hopfield_mf if args.mf else priors.f_in_hopfield
    else:
        prior = priors.f_in_tsodyks_mf if args.mf else priors.f_in_tsodyks

    rnd.seed(args.seed)

    # get samples
    xis0 = hf.get_samples(N, P, args.prior, rho)

    # get a rectified Hopfield connectivity matrix
    if tau is None:
        J, tau = hf.get_J(xis0, nu, None)
    else:
        J = hf.get_J(xis0, nu, tau)

    # find the effective noise and its critical value
    Delta = hf.get_Delta(nu, tau)
    Delta_c = hf.get_Delta_c(args.prior, rho)

    # store
    fname = ("amp_%s_%sN%d_P%d_nu%.3g%s_tau%.3g_init%d_%snumruns%d_s%d.dat" %
             (hf.prior_name(args.prior, rho, True, args.mf),
              ('mf_' if args.mf else ''), N, P, nu,
              (('_rho%g' % rho) if rho is not None else ''),
              tau, args.init, "" if args.no_damping else "adamp_",
              num_runs, args.seed))
    logfile = open(fname, "w", buffering=1)
    header = ('# Rectified Hopfield model with %s prior\n'
              '# N=%d, P=%d, nu=%g, tau=%g, %s\n'
              '# %sDelta=%g, Delta/Delta_c=%g, %s damping, seed=%g\n' %
              (hf.prior_name(args.prior, rho, False, args.mf),
               N, P, nu, tau, amp.init_string(init, False),
               ('' if rho is None else ('rho=%g, ' % rho)), Delta,
               Delta / Delta_c, "no" if args.no_damping else "adaptive",
               args.seed))
    print(header)
    logfile.write(header)

    # construct the Fisher score matrix from the adjacency matrix
    # for the rectified Hopfield channel
    S = hf.get_S(J, nu, tau)

    # solve the inference problem using AMP
    xis, var, F, mse = amp.xx(S, P, prior, args.no_damping,
                              init=init, t_min=20, t_max=100,
                              X0=xis0, verbose=True, rho=rho,
                              logfile=logfile, num_runs=num_runs,
                              diff_converge=1e-6)

    footer = ("# N, P, Delta, Delta_c, mse, F\n# %d, %d, %g, %g, %g, %g\n" %
              (N, P, Delta, Delta_c, mse, F))
    logfile.write(footer)
    print(footer)

    # print("Bye-bye!")


if __name__ == "__main__":
    main()
