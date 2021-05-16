#!/usr/bin/env python3
"""
Tests for the matfact library.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180223
"""

import itertools

import numpy as np
import numpy.random as rnd

import unittest

import amp


def betheFxxFlorentExplicit(logZ, A, B, x, var, S, deltaInv):
    """
    Returns the Bethe free energy per spin in the Bayes optimal case, without
    the minus sign, i.e. big is good.

    Parameters
    ----------
    logZ : (N)
    A : (r, r)
    B : (N, r)
    x : (N, r)
    var : (r, r)
    S : (N, N)
        Fisher matrix of the problem

    Returns
    -------
    F :
        The Bethe Free energy for the given A, B
    """
    N = x.shape[0]

    bethe = 0

    bethe += N / 2 * np.trace(A @ var)
    for i in range(N):
        bethe += logZ[i] - np.dot(B[i], x[i]) + 0.5 * x[i] @ A @ x[i]
        bethe -= 0.5 * deltaInv * x[i] @ var @ x[i]

    for i in range(N):
        for j in range(N):
            bethe -= 0.25 / N * deltaInv * \
                np.trace(np.outer(x[i], x[i]) @ np.outer(x[j], x[j]))

    for i in range(N):
        sum_j = 0
        for j in range(N):
            sum_j += S[i, j] * x[j]
        bethe += 0.5 / np.sqrt(N) * x[i] @ sum_j

    return (bethe / N, 0, 0, 0, 0)


def betheFxxThibaultExplicit(logZ, A, B, x, var, S, deltaInv):
    """
    Returns the Bethe free energy per spin in the Bayes optimal case, without
    the minus sign, i.e. big is good.

    Parameters
    ----------
    logZ : (N)
    A : (r, r)
    B : (N, r)
    x : (N, r)
    var : (r, r)
    S : (N, N)
        Fisher matrix of the problem

    Returns
    -------
    F :
        The Bethe Free energy for the given A, B
    """
    N = x.shape[0]

    bethe = 0

    for i in range(N):
        bethe += (logZ[i] - np.dot(B[i], x[i])
                  + 0.5 * np.trace(A @ (np.outer(x[i], x[i]) + var)))
        for j in range(N):
            bethe += 1/2 / np.sqrt(N) * S[i, j] * np.dot(x[i], x[j])
            bethe -= (1/4 / N * S[i, j]**2 *
                      np.trace(np.outer(x[i], x[i]) @ np.outer(x[j], x[j])))
            bethe -= (1/2 / N * S[i, j]**2 * np.trace(var @ var))

    return (bethe / N, 0, 0, 0, 0)


def betheFxxThibault(logZ, A, B, x, var, S, deltaInv):
    N = x.shape[0]
    bethe = np.sum(logZ)
    bethe += 0.5 * N * np.trace(A @ var)
    bethe += 0.5 * np.trace(A @ (x.T @ x))
    bethe -= np.trace(B @ x.T)

    bethe -= 0.5 * deltaInv * np.trace((x.T @ x) @ var)

    bethe += 0.5 / np.sqrt(N) * np.sum(((x @ x.T)*S))
    bethe -= deltaInv/(4 * N) * np.trace((x.T @ x) @ (x.T @ x))

    return (bethe / N, 0, 0, 0, 0)


class AMPTests(unittest.TestCase):
    def test_betheF(self):
        N = 30
        r = 7

        logZ = rnd.randn(N)
        A = rnd.randn(r, r)
        A = 0.5 * (A + A.T)
        B = rnd.randn(N, r)
        x = rnd.randn(N, r)
        var = rnd.randn(r, r)
        var = 0.5 * (var + var.T)
        S = rnd.randn(N, N)
        S = 0.5 * (S + S.T)

        deltaInv = amp.delta_inv(S)

        self.assertAlmostEquals(betheFxxFlorentExplicit(logZ, A, B, x, var, S, deltaInv)[0],
                                amp.betheFxx(logZ, A, B, x, var, S, deltaInv)[0],
                                msg="Bethe free energy incorrectly implemented")

    def test_mse_sbm(self):
        N = 5000
        r = 2
        # all communities have equal a priori prob.
        g = np.concatenate([i * np.ones(round(N/r))
                            for i in range(r)]).astype(int)
        # shuffle the groups (not strictly necessary)
        rnd.shuffle(g)
        X0 = np.eye(r)[g]

        x = X0 + .5 * rnd.rand(N, r)
        x = x / np.sum(x, axis=1)[:, None]

        mse_explicit = 0
        square = square0 = overlap = 0
        for n in range(N):
            square += 1 / N * np.sum(x[n]**2)
            square0 += 1 / N * np.sum(X0[n]**2)
            overlap += 1 / N * np.sum(np.dot(X0[n], x[n]))
            mse_explicit += 1/N * np.sum((x[n] - X0[n])**2)

        self.assertTrue(np.allclose(amp.get_Q(X0), 1/r * np.eye(r)),
                        msg='self-overlap of truth does not match prior dist.')
        self.assertAlmostEqual(np.trace(amp.get_Q(X0)), square0,
                               msg='error computing Q0')
        self.assertAlmostEqual(np.trace(amp.get_Q(x)), square,
                               msg='error computing self-overlap')
        self.assertAlmostEqual(np.trace(amp.get_M(X0, x)), overlap,
                               msg='error computing magnetisation')

        mse = np.trace(amp.get_Q(X0) + amp.get_Q(x) - 2*amp.get_M(X0, x))

        self.assertAlmostEqual(mse_explicit, mse,
                               msg='wrong mse')

        #  print('\n(!) Nishimori condition M=Q %s violated!' %
        #   ('not' if np.allclose(amp.get_M(X0, x), amp.get_Q(x)) else 'is'))

        #  mse_bayes = np.trace(amp.get_Q(X0) - amp.get_M(X0, x))
        #  self.assertAlmostEqual(mse_explicit, mse_bayes,
        #                         msg='wrong Bayes-optimal mse')

    def test_mse_1_of_r_encoding(self):
        """
        Tests the method to compute the mse between the ground truth and
        estimators for a problem with 1-of-r encoding, such as the SBM, taking
        into account the permutation symmetry of the columns.
        """
        g0 = np.array([1, 2, 1, 0, 0, 2, 1]).astype(int)
        N = g0.shape[0]
        r = np.max(g0) + 1
        X0 = np.eye(r)[g0]

        # create assignment without an error
        g = g0
        X = np.eye(r)[g]
        mse = amp.mse(X0, X)
        self.assertAlmostEqual(0, mse, msg='error computing mse with'
                               'no errors, no permutations')

        # create assignment without an error but permuted columns
        g = g0
        X = np.eye(r)[g]
        cols = np.arange(r)
        rnd.shuffle(cols)
        mse = amp.mse(X0, X[:, cols])
        self.assertAlmostEqual(0, mse, msg='error computing mse with'
                               'no error, permutations')

        # create assignment with two errors
        g = g0
        g[2] = 0
        g[6] = 2
        X = np.eye(r)[g]
        mse = amp.mse(X0, X)
        mse_expected = (2 + 2)/N
        self.assertAlmostEqual(mse_expected, mse, msg='error computing mse with'
                               'two errors, no permutations')

        # same assignment with two errors, permuted
        cols = np.arange(r)
        rnd.shuffle(cols)
        X = np.eye(r)[g]
        mse = amp.mse(X0, X[:, cols])
        mse_expected = (2 + 2)/N
        self.assertAlmostEqual(mse_expected, mse, msg='error computing mse with'
                               'two errors, permutations')

    def test_mse_sign_changes_along_column(self):
        """
        Test motivated by the X for the Hopfield model.
        """
        N = 10
        r = 3

        # store r patterns in N dimensions in an (N, r) matrix
        xis = -1 + 2 * rnd.binomial(1, 0.5, (N, r))
        # change the sign of a single column, i.e. a single pattern
        flipper = np.ones(r)
        flipper[0] = -1
        xis2 = flipper[None, :] * xis

        # first, check that these two matrices give the same X @ X.T matrix
        self.assertTrue(np.allclose(xis @ xis.T, xis2 @ xis2.T))

        # now check that they give the same mse
        self.assertAlmostEqual(amp.mse(xis, xis), 0,
                               msg='MSE not null when comparing same vector')
        self.assertAlmostEqual(amp.mse(xis, -xis), 0,
                               msg='MSE not null after global spin flip')
        self.assertAlmostEqual(amp.mse(xis, xis2), 0,
                               msg='MSE not null when changing sign of a '
                                   'single column')

    def test_mse_consistency(self):
        """
        Tests the consistency of the mse: is every pattern matched to another
        one?
        """
        # generate ground truth
        X0 = np.array([[1, 1],
                       [-1, 1],
                       [-1, 1],
                       [1, -1]])
        # generate an estimate where I flipped just one spin (0, 1); however,
        # now the second pattern is just a global spin flip away from the first
        X = np.array([[1, -1],
                      [-1, 1],
                      [-1, 1],
                      [1, -1]])

        N, r = X0.shape

        mseCorrect = 1 / N * np.sum((X0-X)**2)
        mseWrong = 0  # This would result from wrongly matching the first column
        # of X0 to X[:, 0] and *also* to -X[:, 1], as can happen
        # if one gives the arguments in the wrong order

        self.assertTrue(mseCorrect > 0)
        self.assertAlmostEqual(amp.mse(X0, X), mseCorrect,
                               msg='mse computation inconsistent')

        # Careful: if I give the arguments in the wrong order, matfact
        # underestimates the error!
        self.assertAlmostEqual(amp.mse(X, X0), mseWrong)  # mse estimate = 0

    def test_mse_does_not_concentrate(self):
        """
        Tests that the mse is high even if the estimator concentrates on one of
        several patterns.
        """
        N = 10
        r = 4
        # generate ground truth
        X0 = -1 + 2 * rnd.binomial(1, 0.5, (N, r))

        # generate an estimate where the all three estimates match the first
        # pattern of X0
        X = np.repeat(X0[:, 0], r).reshape(N, r)
        for i in range(r):
            self.assertAlmostEquals(0, np.sum(X0[:, 0] - X[:, i]))

        # brute-force computation of the mse
        mses = []
        for perm in itertools.permutations(range(r)):
            for sign in itertools.product([-1, 1], repeat=r):
                flip = np.array(sign)
                # don't forget the global spin flip |
                #                                  \ /
                #                                   ,
                mses.append(min(1 / N * np.sum((X0 -
                                                flip[None, :]*X[:, perm])**2),
                                1 / N * np.sum((X0 +
                                                flip[None, :]*X[:, perm])**2)))
        expectedMse = min(mses)

        # compare brute-force MSE and the algorithm
        self.assertAlmostEquals(expectedMse, amp.mse(X0, X))

    def test_frac_error(self):
        g0 = np.array([1, 2, 1, 0, 0, 2, 1]).astype(int)
        N = g0.shape[0]
        r = np.max(g0) + 1
        X0 = np.eye(r)[g0]

        # create assignment without an error
        g = g0
        X = np.eye(r)[g]
        frac_err = amp.frac_error(X0, X)
        self.assertAlmostEqual(0, frac_err,
                               msg='error computing fractional error with'
                               'no errors, no permutations')

        # create assignment without an error but permuted columns
        g = g0
        X = np.eye(r)[g]
        cols = np.arange(r)
        rnd.shuffle(cols)
        frac_err = amp.frac_error(X0, X[:, cols])
        self.assertAlmostEqual(0, frac_err,
                               msg='error computing fractional error with'
                               'no error, permutations')

        # create assignment with two errors
        g = g0
        g[2] = 0
        g[6] = 2
        X = np.eye(r)[g]
        frac_err = amp.frac_error(X0, X)
        frac_err_expected = 2/N
        self.assertAlmostEqual(frac_err, frac_err_expected,
                               msg='error computing fractional error with'
                               'two errors, no permutations')

        # same assignment with two errors, permuted
        cols = np.arange(r)
        rnd.shuffle(cols)
        X = np.eye(r)[g]
        frac_err = amp.frac_error(X0, X[:, cols])
        frac_err_expected = 2/N
        self.assertAlmostEqual(frac_err, frac_err_expected,
                               msg='error computing fractional error with'
                               'two errors, permutations')

    def test_M(self):
        N = 10
        r = 3
        r0 = 4

        X0 = rnd.rand(N, r0)
        x = rnd.randn(N, r)

        M_true = np.zeros((r, r0))

        for n in range(N):
            M_true += 1 / N * np.outer(x[n], X0[n])

        M = amp.get_M(X0, x)

        self.assertEqual(r, M.shape[0])
        self.assertEqual(r0, M.shape[1])
        self.assertTrue(np.allclose(M_true, M), 'computed wrong magnetisation')

    def test_Q(self):
        N = 10
        r = 3
        x = rnd.rand(N, r)

        Q_true = np.zeros((r, r))
        for n in range(N):
            Q_true += 1 / N * np.outer(x[n], x[n])

        Q = amp.get_Q(x)

        self.assertTrue(np.allclose(Q_true, Q), 'computed wrong self-overlap')

    def test_Q_sbm(self):
        # generate some dummy data
        N = 5000
        r = 2
        g = np.concatenate([i * np.ones(round(N/r))
                            for i in range(r)]).astype(int)
        # shuffle the groups (not strictly necessary)
        rnd.shuffle(g)
        X0 = np.eye(r)[g]

        self.assertTrue(np.allclose(amp.get_Q(X0), 1/r * np.eye(r)),
                        'wrong self-overlap for planted data')

    def test_Nishimori(self):
        N = 10
        r = 3
        x = rnd.rand(N, r)

        self.assertTrue(np.allclose(amp.get_M(x, x), amp.get_Q(x)),
                        'violating Nishimori conditions')

    def test_Sigma(self):
        N = 10
        r = 3
        var = rnd.rand(N, r, r)

        Sigma_true = np.zeros((r, r))
        for n in range(N):
            Sigma_true += 1 / N * var[n]

        Sigma = amp.get_Sigma(var)

        self.assertTrue(np.allclose(Sigma_true, Sigma),
                        'computed wrong mean variance')

    def test_B_selfaveraging(self):
        # generate some dummy data
        N = 10
        r = 3
        S = 3 * rnd.rand(N, N)
        delta_inv = 2
        x = rnd.rand(N, r)
        x = x / np.sum(x, axis=1)[:, None]
        x_old = rnd.rand(N, r)
        x_old = x_old / np.sum(x_old, axis=1)[:, None]
        var = rnd.rand(r, r)
        var = var + var.T  # make the variance symmetric

        # generate B explicitly according to Eq. 73
        B = np.zeros((N, r))

        for i in range(N):
            B[i] -= delta_inv * var @ x_old[i]
            for k in range(N):
                B[i] += 1 / np.sqrt(N) * S[k, i] * x[k]

        # compare
        B_test = amp.update_B(x, x_old, var, S, delta_inv)
        self.assertTrue(np.allclose(B_test, B),
                        msg='computed wrong B w/ self-averaging')

    def test_B_not_selfaveraging(self):
        N = 4
        r = 3

        S = rnd.randn(N, N)
        x_old = rnd.randn(N, r)
        x = rnd.randn(N, r)
        var = rnd.randn(N, r, r)

        B_new = np.zeros((N, r))
        for i in range(N):
            for k in range(N):
                B_new[i] += (1 / np.sqrt(N) * S[k, i] * x[k]
                             - 1 / N * S[k, i]**2 * var[k] @ x_old[i])

        B_actual = amp.update_B(x, x_old, var, S)
        self.assertTrue(np.allclose(B_new, B_actual),
                        msg="computed wrong B w/out self-averaging")

    def test_A_not_selfaveraging(self):
        N = 4
        r = 3

        S = rnd.randn(N, N)
        x = rnd.randn(N, r)

        A_new = np.zeros((N, r, r))
        for i in range(N):
            for k in range(N):
                A_new[i] += 1 / N * S[k, i]**2 * np.outer(x[k], x[k])

        # 90x slower: A = 1 / N * np.einsum('ki,kl,km->ilm', S**2, x, x)
        # 9x slower: A = 1 / N * \
        #             np.einsum('ki,klm->ilm', S**2, x[:, None] * x[:, :, None])
        # fastest: A = 1 / N * \
        #    np.tensordot((S**2).T, x[:, None] * x[:, :, None], 1)
        A_actual = amp.update_A(x, S)
        self.assertTrue(np.allclose(A_new, A_actual))

    def test_A_selfaveraging(self):
        # generate some dummy data
        N = 10
        r = 5
        x = 4 * rnd.rand(N, r)
        S = rnd.randn(N, N)
        delta_inv = 2

        # compute A explicitly
        A = np.zeros((r, r))
        for k in range(N):
            A += 1 / N * delta_inv * np.outer(x[k], x[k])

        A_test = amp.update_A(x, S, delta_inv)
        self.assertTrue(np.allclose(A_test, A))

    def test_uv_incomplete_ground_truth(self):
        N = 10
        r = 3
        Y = rnd.rand(N, N)
        S = rnd.rand(N, N)
        U = rnd.rand(N, r)
        V = rnd.rand(N, r)

        self.assertRaises(ValueError, amp.uv, Y, S, 3, None, None, 0, U0=None,
                          V0=V)
        self.assertRaises(ValueError, amp.uv, Y, S, 3, None, None, 0, U0=U,
                          V0=None)

    def test_all_differences_mismatched_inputs(self):
        r = 3
        X = 3 * rnd.randn(5, r)
        Y = 3 * rnd.randn(6, r)

        self.assertRaises(ValueError, amp.all_differences, X, Y)

    def test_all_differences(self):
        N = 5
        r1 = 3
        r2 = 4
        X = 3 * rnd.randn(N, r1)
        Y = 3 * rnd.randn(N, r2)

        diffsXX = np.zeros((r1, r1))
        diffsXY = np.zeros((r1, r2))

        for i in range(r1):
            # for XX
            for j in range(r1):
                diffsXX[i, j] = 1 / N * np.sum((X[:, i] - X[:, j])**2)

            # for AB
            for j in range(r2):
                diffsXY[i, j] = 1 / N * np.sum((X[:, i] - Y[:, j])**2)

        self.assertTrue(np.allclose(diffsXX, amp.all_differences(X)),
                        msg='did not correctly compute the difference between '
                            'all the columns of the same matrix.')

        self.assertTrue(np.allclose(diffsXY, amp.all_differences(X, Y)),
                        msg='did not correctly compute the difference between '
                            'all the columns of two matrices.')


if __name__ == '__main__':
    unittest.main()
