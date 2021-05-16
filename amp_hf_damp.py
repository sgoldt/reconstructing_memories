#!/usr/bin/env python

"""
Approximate message passing for pattern reconstruction in the rectified
Hopfield model.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180319

"""

import argparse
import sys

import numpy as np
import numpy.random as rnd
import numpy.linalg as la

from matfact import amp
from matfact import priors

import hopfield as hf

# import importlib.util
# amp_path = "/Users/goldt/Research/software/matfact/matfact/amp.py"
# spec = importlib.util.spec_from_file_location("matfact.amp", amp_path)
# amp = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(amp)

# priors_path = "/Users/goldt/Research/software/matfact/matfact/priors.py"
# spec = importlib.util.spec_from_file_location("matfact.priors", priors_path)
# priors = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(priors)

t_max = 10000

# off_diagonals = []
# off_diagonal = 0

DESCRIPTION = '''
This is AMP for pattern reconstruction in the rectified Hopfield model.
'''

N_DEFAULT = 1000

# def store_A(t, x, var, A, B):
#     global off_diagonal
#     off_diagonal += 0.5 * np.abs(A[0, 1]) + np.abs(A[1, 0])


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-N', '--N', type=int, default=N_DEFAULT,
                        help='number of nodes')
    parser.add_argument('-P', '--P', type=int, default=2,
                        help='number of patterns')
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
    parser.add_argument('--no-damping', action='store_true',
                        help='turn off adaptive damping')
    args = parser.parse_args()

    if args.rho == 1:
        args.rho = None

    (N, P, nu, tau, rho, init) = (args.N, args.P, args.nu, args.tau,
                                  args.rho, args.init)

    # check that the initialisation parameter is valid:
    if init not in amp.INITIALISATIONS:
        raise ValueError("init must be one of 1,2,3,4,5. See matfact help!\n"
                         "Will exit now...")
        exit()
    if rho is not None and (rho < 0 or rho > 1):
        raise ValueError("Sparsity must be a real number between 0, 1."
                         "See the help! Will exit now...")
        exit()

    # find the correct prior
    f_in = None
    if args.prior == hf.PRIOR_HF:
        f_in = priors.f_in_hopfield_mf if args.mf else priors.f_in_hopfield
    else:
        f_in = priors.f_in_tsodyks_mf if args.mf else priors.f_in_tsodyks
    if f_in in [priors.f_in_tsodyks_mf, priors.f_in_tsodyks] and rho is None:
        raise ValueError("need to specify rho for using Tsodyks prior")

    rnd.seed(args.seed)

    # get samples
    X0 = hf.get_samples(N, P, args.prior, rho)

    # get a rectified Hopfield connectivity matrix
    if tau is None:
        J, tau = hf.get_J(X0, nu, None)
    else:
        J = hf.get_J(X0, nu, tau)

    # find the effective noise and its critical value
    Delta = hf.get_Delta(nu, tau)
    Delta_c = hf.get_Delta_c(args.prior, rho)

    # store
    fname = ("amp_%s_%sN%d_P%d_nu%.3g%s_tau%.3g_init%d_s%d.dat" %
             (hf.prior_name(args.prior, rho, True, args.mf),
              ('mf_' if args.mf else ''), N, P, nu,
              (('_rho%g' % rho) if rho is not None else ''),
              tau, args.init, args.seed))
    logfile = open(fname, "w", buffering=1)
    # sys.stdout = logfile
    header = ('# Rectified Hopfield model with %s prior\n'
              '# N=%d, P=%d, nu=%g, tau=%g, %s\n'
              '# %sDelta=%g, Delta/Delta_c=%g, adaptive damping, seed=%g\n' %
              (hf.prior_name(args.prior, rho, args.mf, False),
               N, P, nu, tau, amp.init_string(init, False),
               ('' if rho is None else ('rho=%g, ' % rho)), Delta,
               Delta / Delta_c, args.seed))
    print(header)

    # construct the Fisher score matrix from the adjacency matrix
    # for the rectified Hopfield channel
    S = hf.get_S(J, nu, tau)

    # solve the inference problem using AMP
    converged = False

    # PREPARE
    self_average = True
    deltaInv = amp.delta_inv(S) if self_average else None

    # INITIALISE
    # mean and covariances of the message at i
    B = np.zeros(X0.shape)
    A = np.zeros((P, P))

    # choose a function to compute the error if the ground truth is given
    error_function = amp.mse

    # create a little perturbation
    if args.init == amp.INIT_UNINFORMATIVE:  # default
        # choose initialisation from the prior
        x, var, logZ = f_in(np.zeros((P, P)), np.zeros(X0.shape), rho=rho)
        x += 1e-9 * rnd.randn(*X0.shape)
    elif args.init == amp.INIT_INFORMATIVE:
        # initialise at the (slightly perturbed) solution
        x = X0 + 1e-3 * rnd.randn(*X0.shape)
    elif args.init == amp.INIT_SPECTRAL:
        evals, evecs = la.eigh(S)
        x = evecs[:, -P:]
    elif args.init == amp.INIT_EXACT_TRIVIAL:
        # start exactly at the trivial fixed point
        x = f_in(np.zeros((P, P)), np.zeros(X0.shape), rho=rho)[0]
    elif init == args.INIT_EXACT_INFORMED:
        # start exactly at the solution
        x = X0
    else:
        raise ValueError("initialisation must be between %d and %d." %
                         (min(amp.INITIALISATIONS), max(amp.INITIALISATIONS)))

    x_old = np.zeros(X0.shape)
    # - covariance of the estimate of X
    # - we will only store the average covariance,
    #   which does not prevent us from calculating the covariance of every
    #   estimator
    # - The initialisation of the variance is not important; it will
    #   only be used in the second step, when x_old != 0, and will by then have
    #   been set by the algorithm.
    variancePrior = f_in(np.eye(P, P), np.zeros((N, P)), rho=rho)[1]
    var = np.zeros(variancePrior.shape)

    t = 0
    error = np.zeros(t_max)
    F = np.zeros(t_max)
    old_bethe = 0

    verbose = True
    t_min = 10
    if verbose:
        print("# %2s, %10s, %10s, %10s" %
              ('t', 'error', 'F (Bethe)', 'diff/spin'))

    while True:
        F[t] = amp.betheF(logZ, A, B, x, var, S, deltaInv)
        diff = np.mean(np.abs(x - x_old))

        if X0 is not None:
            error[t] = error_function(X0, x)
            # check we are fulfilling the Nishimori conditions
            # Nc1 = 0  # nishi1(X0, x)
            # Nc2 = nishi2(x, var, variancePrior)

            if verbose:
                print('%4d, %10.4g, %10.4g, %10.4g' %
                      (t, error[t], F[t], diff))
        elif verbose:
            print('%4d, %10.4g, %10.4g' %
                  (t, F[t], diff))

        # do we need to stop yet?
        if t == t_max - 1:
            if verbose:
                print('# Will stop: reached max time')
            break
        elif t > 10 and X0 is not None and error[t] < 1e-6:
            if verbose:
                print('# Will stop: mse < epsilon=%g' % 1e-6)
            converged = True
            break
        elif t > t_min and diff < 1e-6:
            if verbose:
                print('# Will stop: change per spin < %g' % 1e-6)
            converged = True
            break

        # keep the old variables for damping
        B_old = np.copy(B)
        A_old = np.copy(A)

        # compute new means of message s factor -> variable
        B_new = amp.update_B(x, x_old, var, S, deltaInv)

        # compute new variances of messages factor -> variable
        A_new = amp.update_A(x, S, deltaInv)

        # keep the previous estimators before we find new ones using B_new, A_new
        x_old = np.copy(x)

        damp = 1

        if t == 0 or args.no_damping:
            B = B_new
            A = A_new
            x, var, logZ = f_in(A, B, rho=rho)
            old_bethe = amp.betheF(logZ, A, B, x, var, S, deltaInv)
        else:
            goodToGo = False
            while not goodToGo:
                # update according to damping
                B = damp*B_new + (1-damp)*B_old
                A = damp*A_new + (1-damp)*A_old
                x, var, logZ = f_in(A, B, rho=rho)

                bethe = amp.betheF(logZ, A, B, x, var, S, deltaInv)

                if bethe > old_bethe:
                    goodToGo = True
                    old_bethe = bethe
                else:
                    damp /= 2
                    if damp < 1e-2:
                        goodToGo = True
                        old_bethe = bethe
                    else:
                        print("Will try new damping=%g" % damp)

        # minusDKL=np.sum(logZ)+0.5*N*np.trace(A@var)+np.trace(0.5*A@ (x.T @ x))-np.trace(x.T @ B)
        # term_x=-np.trace((x.T @ x)@ var)/(2/ deltaInv)
        # term_xx=np.sum(np.sum(((x@x.T) * S)))/(2*np.sqrt(N))-np.trace((x.T @ x)*(x.T @ x))/(4*N/ deltaInv)
        # free_nrg=(minusDKL+term_x+term_xx)/N
        # print("%g, %g, %g, %g, %g" % (minusDKL / N , term_x / N, term_xx / N, free_nrg, amp.betheF(logZ, A, B, x, var, S, deltaInv)))
        t += 1

    logfile.write("# N, P, Delta, Delta_c, mse, converged\n")
    logfile.write("# %d, %d, %g, %g, %g, %d\n" %
                  (N, P, Delta, Delta_c, error[-1], converged))

    # print("Bye-bye!")


if __name__ == "__main__":
    main()
