"""
Bayes-optimal approximate message passing for low-rank matrix factorisation.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180308
"""

import itertools

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

import scipy.optimize as so

# codes for initialising the AMP estimators
INIT_UNINFORMATIVE = 1  # start with draw from prior, slightly perturbed
INIT_INFORMATIVE = 2    # start with planted solution, slightly perturbed
INIT_SPECTRAL = 3    # start from a spectral initialisation
INIT_EXACT_TRIVIAL = 4  # start exactly at the trivial fixed point
INIT_EXACT_INFORMED = 5  # start exactly at the planted solution
INIT_GAUSSIAN = 6  # start exactly at the planted solution
INITIALISATIONS = [INIT_UNINFORMATIVE,
                   INIT_INFORMATIVE,
                   INIT_SPECTRAL,
                   INIT_EXACT_TRIVIAL,
                   INIT_EXACT_INFORMED,
                   INIT_GAUSSIAN]


def mse(X0: np.ndarray, X: np.ndarray):
    """
    Returns the mean squared error between the ground truth X0 and the estimate
    X as well as the correct permutation of the columns of X.

    **BE WARNED**: This method is *not* symmetric w.r.t. to its arguments, and
    you might underestimate the mse if you change the order of X0 and X. This is
    due to the way we take care of all the possible symmetries in X.

    This method takes into account the following symmetries:
        - global change of sign
        - permutation of the columns
        - change of sign of a single column

    Parameters
    ----------
    X0 : (N, r)
        ground truth spins
    X : (N, r)
        estimate of the spins

    Returns
    -------
    mse :
        lowest mean-squared error over all the permutations of the columns of X,
        sign-flips of the columns, and global sign flips.

    """
    # in the SciPy implementation, if the cost matrix has more columns than
    # rows, as is the case here, then not every column needs to be assigned to a
    # row, as desired.
    costs = np.concatenate((all_differences(X0, X),
                            all_differences(X0, -X)), axis=1)
    row_ind, col_ind = so.linear_sum_assignment(costs)

    return costs[row_ind, col_ind].sum()


def init_estimators(init, X0, S, f_in, f_in_kwargs):
    N, r = X0.shape
    x = None
    if init == INIT_UNINFORMATIVE:  # default
        # choose initialisation from the prior
        x = f_in(np.zeros((r, r)), np.zeros(X0.shape), **f_in_kwargs)[0] \
            + 1e-3 * rnd.randn(*X0.shape)
    elif init == INIT_INFORMATIVE:
        # initialise at the (slightly perturbed) solution
        x = X0 + 1e-3 * rnd.randn(*X0.shape)
    elif init == INIT_SPECTRAL:
        evals, evecs = la.eigh(S)
        x = evecs[:, -r:]
    elif init == INIT_EXACT_TRIVIAL:
        # start exactly at the trivial fixed point
        x = f_in(np.zeros((r, r)), np.zeros(X0.shape), **f_in_kwargs)[0]
    elif init == INIT_EXACT_INFORMED:
        # start exactly at the solution
        x = X0
    elif init == INIT_GAUSSIAN:
        x = rnd.randn(N, r)
    else:
        raise ValueError("initialisation must be between %d and %d." %
                         (min(INITIALISATIONS), max(INITIALISATIONS)))
    return x


def xx(S, r, f_in, no_damping=True, init=INIT_UNINFORMATIVE, epsilon=1e-3,
       t_min=10, t_max=1000, logfile=None,
       X0=None, verbose=True, error_function=mse, callback=None,
       diff_converge=1e-6, self_average=True, num_runs=1,
       **f_in_kwargs):
    """
    Performs AMP for low-rank symmetric matrix factorisation Y = X.T @ X in the
    Bayes-optimal case.

    Parameters:
    -----------
    S : (N, N)
        Fisher information matrix corresponding to Y
    r : scalar
        the rank of the problem. The estimate will have dimensions (N, r)
    f_in : function with two arguments
        implements the prior for the given problem
    no_damping :
        if True, do NOT use adaptive damping.
    init :
        where to initialise the estimators; set to zero by default.
    epsilon :
        error at which to stop the inference; default=1e-3.
    t_min :
        minimal runtime of the algorithm, independently of any other convergence
        criterion
    t_max :
        maximal number of steps; default is 1000.
    X0 : (N, r)
        the planted solution (optional)
    verbose :
        if True, will output continuous status updates. Default is True.
    error_function : function with two arguments
        function to compute the error of the estimator with respect to the
        ground truth. Default: amp.get_mse.
    callback : callable with five arguments
        If provided, this function is called at the beginning of every step of
        the algorithm with the time, the current value of the estimators, their
        variances, and the matrices A and B.
    diff_converge :
        if the change per spin in a step is below this threshold, finish.
    self_average : bool
        if True, assume that the Fisher information is self-averaging.
    **f_in_kwargs :
        keyword-arguments passed on to the prior

    Returns:
    --------
    x : (N, r)
        optimal estimate
    var : (r, r)
        average variance of the N spin estimates
    F :
        Bethe free energy at every time step
    mse :
        mean square reconstruction error at every time-step if X0 was given,
        else None.
    converged :
        True if the algorithm exited due to the estimators converging

    """
    N = S.shape[0]

    # PREPARE
    deltaInv = delta_inv(S) if self_average else None

    # choose a function to compute the error if the ground truth is given
    if X0 is not None and error_function is None:
        error_function = mse

    header = ("# %2s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s, %10s" %
              ('t', 'error', 'F (Bethe)', 'diff/spin', 'damping', "Nishi1",
               "Nishi2", 'minusDKL', 'term2', 'term3', 'term4', 'A'))
    if verbose:
        print(header)

    if logfile is not None:
        logfile.write(header + "\n")

    run = 0
    bestF = -1e5  # highest Bethe free energy reached after convergence
    # parameters of the posterior density with the highest Bethe free energy
    bestA, bestB = np.nan, np.nan
    while run < num_runs:
        # INITIALISE
        # mean and covariances of the message at i
        B = np.zeros(X0.shape)
        A = np.zeros((r, r))

        x = init_estimators(init, X0, S, f_in, f_in_kwargs)
        x_old = np.zeros(x.shape)
        # - covariance of the estimate of X
        # - we will only store the average covariance,
        #   which does not prevent us from calculating the covariance of every
        #   estimator
        # - The initialisation of the variance is not important; it will
        #   only be used in the second step, when x_old != 0,
        #   and will by then have been set by the algorithm.
        variancePrior = f_in(np.eye(r, r), np.zeros((N, r)), **f_in_kwargs)[1]
        var = np.zeros(variancePrior.shape)

        logZ = np.nan  # this value will never be used
        damp = np.nan  # this value will never be used
        error = np.nan
        F = np.nan  # this value will never be used
        t = 0
        x = init_estimators(init, X0, S, f_in, f_in_kwargs)
        B = np.zeros(X0.shape)
        A = np.zeros((r, r))

        while True:
            if t > 0:
                F, minusDKL, term2, term3, term4 = \
                    betheFxx(logZ, A, B, x, var, S, deltaInv)

            diff = np.nan if t == 0 else np.mean(np.abs(x - x_old))
            if callback is not None:
                callback(t, x, var, A, B)

            if X0 is not None:  # compute the error, print the status
                error = error_function(X0, x)

                nc1 = np.nan  # nishi1(X0, x)
                nc2 = np.nan  # nishi2(x, var, variancePrior)
                msg = ("%4d, %10.4g, %10.4g, %10.4g, %10.4g, %10.4g, %10.4g" %
                       (t, error, F, diff,
                        np.nan if t == 0 else damp, nc1, nc2))
                if t > 0:
                    msg += (", %10.4g, %10.4g, %10.4g, %10.4g, %10.4g" %
                            (minusDKL, term2, term3, term4, np.mean(np.diag(A))))
                else:
                    msg += (", %10.4g, %10.4g, %10.4g, %10.4g, %10.4g" %
                            (np.nan, np.nan, np.nan, np.nan, np.nan))
                if verbose:
                    print(msg)
                    # print("%10.4g, " * 10 % (0, *list(X0[39:48, :].flatten())))
                    # print("%10.4g, " * 10 % (0, *list(x[39:48, :].flatten())))
                    # print("%10.4g, " * 10 % (0, *list(X0[9:18, :].flatten())))
                    # print("%10.4g, " * 10 % (0, *list(x[9:18, :].flatten())))
                    # print("%10.4g, " * 10 % (0, *list(X0[19:28, :].flatten())))
                    # print("%10.4g, " * 10 % (0, *list(x[19:28, :].flatten())))
                    # print("\n")
                if logfile is not None:
                    logfile.write(msg + "\n")
            elif verbose:
                print("%4d, %10.4g, %10.4g, %10.4g" %
                      (t, F, diff, np.nan if t == 0 else damp))
            if X0 is None and logfile is not None:
                logfile.write("%4d, %10.4g, %10.4g, %10.4g\n" %
                              (t, F, diff, np.nan if t == 0 else damp))

            # do we need to stop yet?
            if t > t_min and X0 is not None and error < epsilon:
                if verbose:
                    print('# Will stop: mse < epsilon=%g' % epsilon)
                break
            elif t == t_max - 1 or (t > t_min and diff < diff_converge):
                if verbose:
                    print('# Run finished. (diff too small or max time reached)')
                # let's reset
                break

            # keep the old variables for damping
            B_old = np.copy(B)
            A_old = np.copy(A)

            # compute new means of message s factor -> variable
            B_new = update_B(x, x_old, var, S, deltaInv)

            # compute new variances of messages factor -> variable
            A_new = update_A(x, S, deltaInv)

            # keep the previous estimators
            x_old = np.copy(x)

            damp = 1

            if t < 3 or no_damping:
                B = B_new
                A = A_new
                x, var, logZ = f_in(A, B, **f_in_kwargs)
                old_F, _, _, _, _ = betheFxx(logZ, A, B, x, var, S, deltaInv)
            else:
                goodToGo = False
                while not goodToGo:
                    # update according to damping
                    B = damp*B_new + (1-damp)*B_old
                    A = damp*A_new + (1-damp)*A_old
                    x, var, logZ = f_in(A, B, **f_in_kwargs)

                    F, _, _, _, _ = betheFxx(logZ, A, B, x, var, S, deltaInv)

                    if F >= old_F or np.isnan(F):
                        goodToGo = True
                    else:
                        damp /= 2
                        if damp < 1e-4:
                            damp *= 2
                            goodToGo = True
                old_F = F

            t += 1

        # We just finished a run of AMP
        if np.isnan(F):
            # the free energy cannot be computed, e.g. because the prior does
            # not provide the partition function; thus, we break
            bestMSE = error if X0 is not None else np.nan
            bestF = np.nan
            break
        elif F > bestF:  # did it obtain the highest likelihood yet?
            bestF = F
            bestA, bestB = A, B
            bestMSE = error
        # if bestF > 0.1:
        #     break
        else:
            run += 1
            print("# Let's reset")
            logfile.write("# Let's reset\n\n")

    # obtain estimators from the posterior density with the highest free energy
    if not np.isnan(F):
        x, var, logZ = f_in(bestA, bestB, **f_in_kwargs)
    return x, var, bestF, bestMSE if X0 is not None else np.nan


def uv(S, r, f_in_u, f_in_v, damp, init=INIT_UNINFORMATIVE, epsilon=1e-3,
       t_min=10, t_max=1000,
       U0=None, V0=None, verbose=True, error_function=mse, diff_converge=1e-6,
       **f_in_kwargs):
    """
    Performs AMP for low-rank symmetric matrix factorisation Y = U @ V.T. in the
    Bayes-optimal case.

    Parameters:
    -----------
    S : (N, N)
        Fisher information matrix corresponding to Y
    r : scalar
        the rank of the problem. The estimate will have dimensions (N, r)
    f_in_u, f_in_v : each a function with two arguments
        implements the prior for the U and V matrics, respectively.
        (a.k.a. as thresholding function)
    damp :
        damping factor, where damp=0 -> no damping, damp=1 -> no progress
    init :
        where to initialise the estimators; set to zero by default.
    epsilon :
        error at which to stop the inference; default=1e-3.
    t_min :
        minimal runtime of the algorithm, independently of any other convergence
        criterion
    t_max :
        maximal number of steps; default is 1000.
    U0, V0 : (N, r)
        the planted solution for U and V (optional)

    Returns:
    --------
    u : (N, r)
        optimal estimates of U
    v : (M, r)
        optimal estimates of V
    var_u, var_v : (r, r)
        average variance of the estimates of U and V, resp.
    F :
        Bethe free energy at every time step
    error :
        error as computed by the function error_function at every time-step if
        U0, V0 were given, else None.
    converged :
        True if the algorithm exited due to the estimators converging

    """
    converged = False
    # check the arguments
    if (U0 is None and V0 is not None) or (U0 is not None and V0 is None):
        raise ValueError('Must either give the ground truth for both U, V or '
                         'none of them.')

    # PREPARE
    deltaInv = delta_inv(S)

    # INITIALISE
    # mean and covariances of the message at i
    B_u = np.zeros(U0.shape)
    A_u = np.zeros((r, r))
    B_v = np.zeros(V0.shape)
    A_v = np.zeros((r, r))

    if init == INIT_UNINFORMATIVE:  # default
        # choose initialisation from the prior
        u = f_in_u(np.zeros((r, r)), np.zeros(U0.shape), **f_in_kwargs)[0] \
            + 1e-3 * rnd.randn(*U0.shape)
        v = f_in_v(np.zeros((r, r)), np.zeros(V0.shape), **f_in_kwargs)[0] \
            + 1e-3 * rnd.randn(*V0.shape)
    elif init == INIT_INFORMATIVE:
        # initialise at the (slightly perturbed) solution
        u = U0 + 1e-3 * rnd.randn(*U0.shape)
        v = V0 + 1e-3 * rnd.randn(*V0.shape)
    elif init == INIT_EXACT_TRIVIAL:
        # start exactly at the trivial fixed point
        u = f_in_u(np.zeros((r, r)), np.zeros(U0.shape), **f_in_kwargs)[0]
        v = f_in_v(np.zeros((r, r)), np.zeros(V0.shape), **f_in_kwargs)[0]
    elif init == INIT_EXACT_INFORMED:
        # start exactly at the solution
        u = U0
        v = V0
    else:
        raise ValueError("initialisation must be between %d and %d." %
                         (min(INITIALISATIONS), max(INITIALISATIONS)))

    u_old = np.zeros(U0.shape)
    v_old = np.zeros(V0.shape)
    # - covariance of the estimate of X
    # - we will only store the average covariance,
    #   which does not prevent us from calculating the covariance of every
    #   estimator
    # - The initialisation of the variance is not important; it will
    #   only be used in the second step, when x_old != 0, and will by then have
    #   been set by the algorithm.
    var_u = np.zeros((r, r))
    var_v = np.zeros((r, r))
    variancePrior_u = f_in_u(np.zeros((r, r)), np.zeros(U0.shape),
                             **f_in_kwargs)[1]
    variancePrior_v = f_in_v(np.zeros((r, r)), np.zeros(V0.shape),
                             **f_in_kwargs)[1]

    t = 0
    error = np.zeros(t_max)
    F = np.zeros(t_max)

    if verbose:
        print("# %2s, %8s, %8s, %8s, %8s, %8s, %8s, %8s" %
              ('t', 'mse', 'F (Bethe)', 'diff/spin',
               'Nish1(U)', 'Nish2(U)', 'Nish1(V)', 'Nish2(V)'))
    while True:
        # no need to change community labels for the free energy
        # F[t] = get_Bethe_F(A, B, x, var, S)
        diff = 1/2*(np.mean(np.abs(u - u_old)) + np.mean(np.abs(v - v_old)))

        if U0 is not None:
            # enough to check for the presence of only one ground truth matrix
            error[t] = error_function(U0, u) + error_function(V0, v)
            # check we are fulfilling the Nishimori conditions
            Nc_u_1 = 0  # nishi1(U0, u)
            Nc_u_2 = nishi2(u, var_u, variancePrior_u)
            Nc_v_1 = 0  # nishi1(V0, v)
            Nc_v_2 = nishi2(v, var_v, variancePrior_v)

            if verbose:
                print('%4d, %8.4g, %8.4g, %8.4g, %8.4g, %8.4g, %8.4g, %8.4g' %
                      (t, error[t], F[t], diff,
                       Nc_u_1, Nc_u_2 if t > 0 else 0,
                       Nc_v_1, Nc_v_2 if t > 0 else 0))
        elif verbose:
            print('%4d, %8.4g, %8.4g' %
                  (t, F[t], diff))

        # do we need to stop yet?
        if t == t_max - 1:
            if verbose:
                print('# Will stop: reached max time')
            break
        elif U0 is not None and error[t] < epsilon:
            if verbose:
                print('# Will stop: mse < epsilon=%g' % epsilon)
            converged = True
            break
        elif t > t_min and diff < diff_converge:
            if verbose:
                print('# Will stop: change per spin < %g' % diff_converge)
            converged = True
            break

        # keep the old variables for damping
        B_u_old = np.copy(B_u)
        A_u_old = np.copy(A_u)
        A_v_old = np.copy(A_v)
        B_v_old = np.copy(B_v)

        # compute new means of message
        B_u_new = update_B(v, u_old, var_v, S.T, deltaInv)
        B_v_new = update_B(u,     v, var_u,   S, deltaInv)

        # compute new variances of messages
        A_u_new = update_A(u, S, deltaInv)
        A_v_new = update_A(v, S, deltaInv)

        # update according to damping
        B_u = (1 - damp)*B_u_new + damp*B_u_old
        A_u = (1 - damp)*A_u_new + damp*A_u_old
        B_v = (1 - damp)*B_v_new + damp*B_v_old
        A_v = (1 - damp)*A_v_new + damp*A_v_old

        # update mean and covariance of the estimator x
        u_old = np.copy(u)
        v_old = np.copy(v)
        u, var_u = f_in_u(A_u, B_u, **f_in_kwargs)
        v, var_v = f_in_v(A_v, B_v, **f_in_kwargs)

        t += 1

    return (u, v, var_u, var_v, F[:t+1],
            (error[:t+1] if U0 is not None else None), converged)


def update_A(x, S, deltaInv=None):
    """
    Parameters:
    -----------
    x : (N, r)
    S : (N, N)
    deltaInv : double or None
        if a value is given, assume that we can replace S**2 by its mean w/out
        changing the leading order of the quantities.
    """
    N = S.shape[0]
    if deltaInv is None:
        xxT = x[:, None] * x[:, :, None]
        return 1 / N * np.tensordot((S**2).T, xxT, 1)
    else:
        return 1 / N * deltaInv * x.T @ x


def update_B(x, x_old, var, S, deltaInv=None):
    """
    Parameters:
    -----------
    x : (N, r)
    x_old : (N, r)
    var : (r, r)
    S : (N, N)
    N : int
    delta_inv : double or None
        if a value is given, assume that we can replace S**2 by its mean w/out
        changing the leading order of the quantities.
    """
    N = S.shape[0]
    if deltaInv is None:
        S2var = np.tensordot((S**2).T, var, 1)
        return (1 / np.sqrt(N) * S.T @ x
                - 1 / N * np.squeeze(S2var @ x_old[:, :, None]))
    else:
        return 1 / np.sqrt(N) * S.T @ x - deltaInv * x_old @ var


def all_differences(A, B=None):
    """
    For two Nxr matrices, returns the rxr matrix that contains the squared
    difference summed over N for all the combinations of columns.

    Such a matrix can then be used to find the minimal mse over all permutations
    of rows of an estimate in an AMP problem over permutations of the columns of
    the estimator.

    Parameters
    ----------
    A : (N, r1)
    B : (N, r2) (optional)
        if None is given, the algorithm computes all_differences(A, A).

    Returns
    -------
    diffs : (r1, r2)
        diff[i, j] is 1 / N * (A[:, i] - B[:, j])**2
    """
    N = A.shape[0]

    if B is None:
        B = A
    elif A.shape[0] != B.shape[0]:
        raise ValueError("number of rows of  A, B, must be equal.")

    diffs = (np.einsum('ij,ij->j', A, A)[:, None] +
             np.einsum('ij,ij->j', B, B) - 2*np.dot(A.T, B))

    return diffs / N


def frac_error(X0: np.ndarray, X: np.ndarray):
    """
    Returns the fractional error for symmetric matrix factorisation where the
    matrix X can be interpreted as a 1-of-r encoding, for example of a
    community.

    Parameters
    ----------
    X0 : (N, r)
        ground truth spins
    X : (N, r)
        estimate of the spins

    Returns
    -------
    frac_err :
        lowest fractional error over all the permutaitons of the columns of X
    """
    N, r = X0.shape
    # correct group assignments
    g0 = np.argmax(X0, axis=1)  # current group assignment

    frac_errors = []

    for perm in itertools.permutations(range(r)):
        g = np.argmax(X[:, perm], axis=1)  # current group assignment
        frac_errors.append(1 / N * np.count_nonzero(g0 - g))

    # get the lowest mse and its index
    return min(frac_errors)


def nishi1(X0, x):
    '''
    Numerically checks the first Nishimori condition.

    Parameters:
    -----------
    X0 : (N, r0)
        planted solution
    x : (N, r)
        estimate

    '''
    r = x.shape[1]

    nishis = []

    for p in itertools.permutations(range(r)):
        # check for the global spin flip, too!
        nishis.append(min(np.sum(np.abs(get_Q(x) - get_M(X0, x[:, p]))),
                          np.sum(np.abs(get_Q(x) - get_M(X0, -x[:, p])))))

    return min(nishis)


def nishi2(x, var, variancePrior):
    '''
    Numerically checks the second Nishimori condition.

    Parameters:
    -----------
    x : (N, r)
        estimate of the means
    var : (r, r)
        current average estimate of the variance
    variancePrior : (r, r)
        covariance of the prior distribution
    '''
    return np.sum(np.abs(get_Q(x) + var - variancePrior))


def get_M(X0, x):
    """
    Returns the magnetisation order parameter.

    Parameters:
    -----------
    X0 : (N, r0)
        planted solution
    x : (N, r)
        estimate

    Returns:
    -------
    M : (r, r0)
    """
    N = x.shape[0]
    return 1 / N * x.T @ X0


def get_Q(x):
    """
    Returns the self-overlap order parameter.

    Parameters:
    -----------
    x : (N, r)
       estimate
    Returns:
    -------
    Q : (r, r)
    """
    N = x.shape[0]
    return 1 / N * x.T @ x


def get_Sigma(var):
    """
    Returns the Sigma order parameter, which is the average of the variances of
    the estimators.

    Parameters:
    -----------
    var : (N, r, r)

    Returns:
    -------
    Sigma : (r, r)
    """
    return np.mean(var, axis=0)


def delta_inv(S: np.ndarray):
    """
    Returns the 'empirical' inverse Fisher information, Eq. 75 of Lesieur 2017
    """
    N = S.shape[0]
    return 2 / N**2 * np.triu(S**2, k=1).sum()


def init_string(init, short=True):
    """
    Returns a string representation of the given initialisation

    Parameters:
    -----------
    init : int
        initialisation code
    short : bool
        if True, only a two-letter code for the initialisation is given.
        Default: True.

    Raises:
    -------
    ValueError :
        if the given initialisation code is not one of the predefined ones.
    """
    if init == INIT_UNINFORMATIVE:
        return ("ui" if short else "uninformative initialisation")
    elif init == INIT_INFORMATIVE:
        return ("ii" if short else "informative initialisation")
    elif init == INIT_SPECTRAL:
        return ("si" if short else "spectral initialisation")
    elif init == INIT_GAUSSIAN:
        return ("gi" if short else "gaussian initialisation")
    else:
        raise ValueError("Illegal initialisation code given!")


def betheFxx(logZ, A, B, x, var, S, deltaInv):
    N = x.shape[0]

    minusDKL = np.sum(logZ) + 0.5 * N * np.trace(A @ var) \
        + np.trace(0.5 * A @ x.T @ x) - np.trace(x.T @ B)
    term2 = -0.5 * deltaInv * np.trace(x.T @ x @ var)
    term4 = 0.5 / np.sqrt(N) * np.sum((x @ x.T) * S)
    term3 = - 0.25 / N * deltaInv * np.trace((x.T @ x) @ (x.T @ x))
    bethe = (minusDKL + term2 + term3 + term4)

    return (bethe / N, minusDKL / N, term2 / N, term3 / N, term4 / N)
