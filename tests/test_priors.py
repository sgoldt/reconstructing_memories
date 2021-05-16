#!/usr/bin/env python3
"""
Tests for the AMP priors.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180323
"""

import itertools
import numpy as np
import numpy.random as rnd

import unittest

import priors


class TsodyksTests(unittest.TestCase):
    def test_pX(self):
        rho = 0.5 * rnd.rand()
        p_pos = rho  # pr(x_i = 1-rho)
        p_neg = 1 - rho  # pr(x_i = -rho)

        x = np.array([-rho, -rho, -rho, 1-rho, 1-rho, -rho, -rho, 1-rho])
        p = p_pos**3 * p_neg**5

        self.assertAlmostEqual(p,
                               priors.pX_tsodyks(x, rho),
                               msg='Wrong prior for Tsodyks-like model, N=1')

        x = np.array([[1-rho, 1-rho, 1-rho, -rho, -rho, 1-rho, 1-rho, -rho],
                      [1-rho, 1-rho, 1-rho, -rho, -rho, 1-rho, 1-rho, 1-rho],
                      [1-rho, 1-rho, 1-rho, -rho, 1-rho, 1-rho, 1-rho, 1-rho]])
        p = [p_pos**5 * p_neg**3,
             p_pos**6 * p_neg**2,
             p_pos**7 * p_neg]

        self.assertTrue(np.allclose(p, priors.pX_tsodyks(x, rho)),
                        msg='Wrong prior for Tsodyks-like model, N=3')

    def test_f_in_multiple(self):
        '''
        Check the implementation of the threshold function for N distributions.
        '''
        # generate some dummy data
        N = 5
        r = 3
        A = 3*rnd.rand(r, r)
        B = rnd.randn(N, r)
        rho = 0.5 * rnd.rand()

        # First, compute explicitly the means...
        means0 = np.zeros((N, r))
        for n in range(N):
            for i in itertools.product([-rho, 1-rho], repeat=r):
                xi = np.array(i)
                means0[n] += xi*self.__W(xi, A, B[n], rho)

        # and the covariance for each of the N distributions
        var0 = np.zeros((N, r, r))  # will later average over first dimension
        for n in range(N):
            for i in itertools.product([-rho, 1-rho], repeat=r):
                xi = np.array(i)
                var0[n] += np.outer(xi, xi)*self.__W(xi, A, B[n], rho)
            var0[n] -= np.outer(means0[n], means0[n])
        var0 = np.mean(var0, axis=0)

        # and the partition function
        Z0 = np.zeros(N)
        for n in range(N):
            for i in itertools.product([-rho, 1-rho], repeat=r):
                xi = np.array(i)
                pxi = priors.pX_tsodyks(xi, rho)
                Z0[n] += pxi * np.exp(B[n] @ xi - xi.T @ A @ xi / 2)
        logZ0 = np.log(Z0)

        means, var, logZ = priors.f_in_tsodyks(A, B, rho)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')
        self.assertTrue(np.allclose(logZ, logZ0), msg='ln(Z) wrong')

    def test_f_in_single(self):
        '''
        Check the implementation of the prior for a single distribution.
        '''
        # generate some dummy data
        r = 3
        A = 3*rnd.rand(r, r)
        B = rnd.randn(r)
        rho = 0.5 * rnd.rand()

        # First, compute explicitly the mean...
        means0 = np.zeros((r))
        for i in itertools.product([-rho, 1-rho], repeat=r):
            xi = np.array(i)
            means0 += xi*self.__W(xi, A, B, rho)

        # and the covariance for the distributions
        var0 = np.zeros((r, r))
        for i in itertools.product([-rho, 1-rho], repeat=r):
            xi = np.array(i)
            var0 += np.outer(xi, xi)*self.__W(xi, A, B, rho)
        var0 -= np.outer(means0, means0)

        means, var, logZ = priors.f_in_tsodyks(A, np.array([B]), rho)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')

    def test_W_normalisation(self):
        N = 4
        r = 5
        A = 3*rnd.rand(r, r)
        B = rnd.randn(N, r)
        rho = 0.5 * rnd.rand()

        Z = 0
        for i in itertools.product([-rho, 1-rho], repeat=r):
            xi = np.array(i)
            Z += self.__W(xi, A, B[2], rho)

        self.assertAlmostEqual(1, Z)

    def __W(self, x, A, B, rho):
        """
        Distribution for a single mean B and single covariance matrix A.

        Parameters:
        -----------
        x : (r,)
            single input, $x\in\mathbb{R}^r$
        A : (r, r)
        B : (r,)
        rho : scalar
            0 < rho < 1

        Returns
        -------
        scalar
        """
        r = x.shape[0]

        Z = 0
        for i in itertools.product([-rho, 1-rho], repeat=r):
            xi = np.array(i)
            pxi = priors.pX_tsodyks(xi, rho)
            Z += pxi * np.exp(B @ xi - xi.T @ A @ xi / 2)

        px = priors.pX_tsodyks(x, rho)
        return 1 / Z * px * np.exp(B @ x - x.T @ A @ x / 2)


class HopfieldTests(unittest.TestCase):
    def test_f_in_multiple(self):
        '''
        Check the implementation of the threshold function for N distributions.
        '''
        # generate some dummy data
        N = 5
        r = 3
        A = 3*rnd.rand(r, r)
        B = rnd.randn(N, r)

        # First, compute explicitly the means...
        means0 = np.zeros((N, r))
        for n in range(N):
            for i in itertools.product([-1, 1], repeat=r):
                xi = np.array(i)
                means0[n] += xi*self.__W(xi, A, B[n])

        # and the covariance for each of the N distributions
        var0 = np.zeros((N, r, r))  # will later average over first dimension
        for n in range(N):
            for i in itertools.product([-1, 1], repeat=r):
                xi = np.array(i)
                var0[n] += np.outer(xi, xi)*self.__W(xi, A, B[n])
            var0[n] -= np.outer(means0[n], means0[n])
        var0 = np.mean(var0, axis=0)

        Z0 = np.zeros(N)
        for n in range(N):
            for i in itertools.product([-1, 1], repeat=r):
                xi = np.array(i)
                Z0[n] += 1 / 2**r * np.exp(B[n] @ xi - xi.T @ A @ xi / 2)
        logZ0 = np.log(Z0)

        means, var, logZ = priors.f_in_hopfield(A, B)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')
        self.assertTrue(np.allclose(logZ, logZ0), msg='ln Z wrong')

    def test_f_in_single(self):
        '''
        Check the implementation of the prior for a single distribution.
        '''
        # generate some dummy data
        r = 3
        A = 3*rnd.rand(r, r)
        B = rnd.randn(r)

        # First, compute explicitly the mean...
        means0 = np.zeros((r))
        for i in itertools.product([-1, 1], repeat=r):
            xi = np.array(i)
            means0 += xi*self.__W(xi, A, B)

        # and the covariance for the distributions
        var0 = np.zeros((r, r))
        for i in itertools.product([-1, 1], repeat=r):
            xi = np.array(i)
            var0 += np.outer(xi, xi)*self.__W(xi, A, B)
        var0 -= np.outer(means0, means0)

        means, var, logZ = priors.f_in_hopfield(A, np.array([B]))

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')

    def test_W_normalisation(self):
        N = 4
        r = 5
        A = 3*rnd.rand(r, r)
        B = rnd.randn(N, r)

        Z = 0
        for i in itertools.product([-1, 1], repeat=r):
            xi = np.array(i)
            Z += self.__W(xi, A, B[2])

        self.assertAlmostEqual(1, Z)

    def __W(self, x, A, B):
        """
        Distribution for a single mean B and single covariance matrix A.

        Parameters:
        -----------
        x : (r,)
            $x\in\{ \pm 1\}^r$
        A : (r, r)
        B : (r,)

        Returns
        -------
        scalar
        """
        r = x.shape[0]

        Z = 0
        for i in itertools.product([-1, 1], repeat=r):
            xi = np.array(i)
            Z += np.exp(B @ xi - xi.T @ A @ xi / 2)
        return 1 / Z * np.exp(B @ x - x.T @ A @ x / 2)

    def test_pX_sparse(self):
        rho = rnd.rand()
        p0 = 1 - rho  # pr(x_i = 0)
        p1 = rho / 2  # pr(x_i = +-1)

        x = np.array([0, 0, 0, -1, -1, 0, 0, 1])
        p = p0**3 * p1**2 * p0**2 * p1

        self.assertAlmostEqual(p,
                               priors.pX_hopfield_sparse(x, rho),
                               msg='Wrong prior for sparse HF model, N=1')

        x = np.array([[0, 0, 0, -1, -1, 0, 0, 1],
                      [0, 1, 1, -1, -1, 0, 0, 1],
                      [0, 0, 1, -1, 0, 0, 0, 0]])
        p = [p0**3 * p1**2 * p0**2 * p1,
             p0 * p1**4 * p0**2 * p1,
             p0**2 * p1**2 * p0**4]
        self.assertTrue(np.allclose(p, priors.pX_hopfield_sparse(x, rho)),
                        msg='Wrong prior for sparse HF model, N=3')

    def test_pX_sparse_normalisation(self):
        rho = rnd.rand()

        for r in [1, 4]:
            Z = 0
            for x in itertools.product([-1, 0, 1], repeat=r):
                xi = np.array(x)
                Z += priors.pX_hopfield_sparse(xi, rho)
            error_msg = ('pX for sparse Hopfield not normalised for r=%d' % r)
            self.assertAlmostEqual(Z, 1, msg=error_msg)

    def test_f_in_sparse_multiple(self):
        """
        Check the implementation of the threshold function with N distributions
        for the sparse Hopfield model, a.k.a. Rademacher-Bernoulli.
        """
        # generate some dummy data
        N = 5
        r = 3
        rho = rnd.rand()
        A = 3*rnd.rand(r, r)
        B = rnd.randn(N, r)

        # First, compute explicitly the means...
        means0 = np.zeros((N, r))
        for n in range(N):
            for i in itertools.product([-1, 0, 1], repeat=r):
                xi = np.array(i)
                means0[n] += xi*self.__W_sparse(xi, A, B[n], rho)

        # and the covariance for each of the N distributions
        var0 = np.zeros((N, r, r))  # will later average over first dimension
        for n in range(N):
            for i in itertools.product([-1, 0, 1], repeat=r):
                xi = np.array(i)
                var0[n] += np.outer(xi, xi) * \
                    self.__W_sparse(xi, A, B[n], rho)
            var0[n] -= np.outer(means0[n], means0[n])
        var0 = np.mean(var0, axis=0)

        Z0 = np.zeros(N)
        for n in range(N):
            for i in itertools.product([-1, 0, 1], repeat=r):
                xi = np.array(i)
                pxi = priors.pX_hopfield_sparse(xi, rho)
                Z0[n] += pxi * np.exp(B[n] @ xi - xi.T @ A @ xi / 2)
        logZ0 = np.log(Z0)

        means, var, logZ = priors.f_in_hopfield(A, B, rho)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')
        self.assertTrue(np.allclose(logZ, logZ0), msg='ln Z wrong')

    def test_f_in_fast_sparse_Ndist(self):
        """
        Check the implementation of the fast threshold function f_in with N
        distributions for the sparse Hopfield model, a.k.a.
        Rademacher-Bernoulli.
        """
        # generate some dummy data
        N = 5
        r = 3
        rho = rnd.rand()
        # only assumption of the fast implementation: diagonal A!
        A = np.diag(rnd.rand(r))
        B = rnd.randn(N, r)

        # First, compute explicitly the means...
        means0 = np.zeros((N, r))
        for n in range(N):
            for i in itertools.product([-1, 0, 1], repeat=r):
                xi = np.array(i)
                means0[n] += xi*self.__W_sparse(xi, A, B[n], rho)

        # and the covariance for each of the N distributions
        var0 = np.zeros((N, r, r))  # will later average over first dimension
        for n in range(N):
            for i in itertools.product([-1, 0, 1], repeat=r):
                xi = np.array(i)
                var0[n] += np.outer(xi, xi) * \
                    self.__W_sparse(xi, A, B[n], rho)
            var0[n] -= np.outer(means0[n], means0[n])
        var0 = np.mean(var0, axis=0)

        means, var, logZ = priors.f_in_hopfield(A, B, rho, diagonalA=True)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')

    def test_W_sparse_normalisation(self):
        N = 4
        r = 5
        rho = rnd.rand()
        A = 3*rnd.rand(r, r)
        B = rnd.randn(N, r)

        Z = 0
        for i in itertools.product([-1, 0, 1], repeat=r):
            xi = np.array(i)
            Z += self.__W_sparse(xi, A, B[2], rho)

        self.assertAlmostEqual(1, Z)

    def __W_sparse(self, x, A, B, rho):
        """
        Distribution for a single mean B and single covariance matrix A of the
        sparse Hopfield model.

        Parameters:
        -----------
        x : (r,)
            $x\in\{ 0, \pm 1\}^r$
        A : (r, r)
        B : (r,)

        Returns
        -------
        scalar
        """
        r = x.shape[0]

        Z = 0
        for i in itertools.product([-1, 0, 1], repeat=r):
            xi = np.array(i)
            pxi = priors.pX_hopfield_sparse(xi, rho)
            Z += pxi * np.exp(B @ xi - xi.T @ A @ xi / 2)

        px = priors.pX_hopfield_sparse(x, rho)
        return 1 / Z * px * np.exp(B @ x - x.T @ A @ x / 2)

    def test_mf(self):
        A = np.array([[1.99231813, 0.02768281, 0.69547137],  # 3*rand(r, r)
                      [2.60750208, 2.38842074, 2.0617683],
                      [1.84386443, 2.55858706, 0.40241918]])
        B = np.array([[-1.61031, 1.54639381, 1.50972913],
                      [1.09592879, 0.22937869, 1.27963542],
                      [0.94992942, -0.91819765, -1.07318164],
                      [0.45385694, 1.08005699, 0.98716607]])

        # the correct output
        mean0 = np.array([[-0.96946819,  0.99191237,  0.64072511],
                          [0.40610468, -0.99374883,  0.99572873],
                          [0.67425056, -0.99631736,  0.22864234],
                          [-0.21471093, -0.3645039,  0.98070492]]),
        cov0 = np.array([[0.59862398, -0.00291551, -0.0518035],
                         [-0.27461796,  0.22576543, -0.01281132],
                         [-0.13734373, -0.01589844,  0.39598405]])

        mean, cov, _ = priors.f_in_hopfield_mf(A, B)
        self.assertTrue(np.allclose(mean, mean0),
                        msg='mean is wrong for mean-field Hopfield prior')
        self.assertTrue(np.allclose(cov, cov0),
                        msg='cov is wrong for mean-field Hopfield prior')


class JointlyGaussBernoulliTests(unittest.TestCase):
    def test_f_in_single(self):
        """
        Tests the threshold function for the Gauss-Bernoulli prior for a single
        distribution.
        """
        # data from a Mathematica notebook
        A = np.array([[2, 0.7], [0.7, 1.5]])
        B = np.array([[1.2, -0.7]])
        rho = 0.3

        mean0 = np.array([[0.100438, -0.0846095]])
        var0 = np.array([[0.111863, -0.0537709],
                         [-0.0537709, 0.114663]])

        mean, var = priors.f_in_jointly_gauss_bernoulli(A, B, rho)

        self.assertTrue(np.allclose(mean, mean0),
                        msg='mean wrong')
        self.assertTrue(np.allclose(var, var0),
                        msg='variance wrong')

    def test_f_in_multiple(self):
        """
        Tests the threshold function for the Gauss-Bernoulli prior for a many
        distributions.
        """
        # data from a Mathematica notebook
        A = np.array([[2, 0.7], [0.7, 1.5]])
        B = np.array([[1.2, -0.7],
                      [-0.33, -0.6],
                      [1.13, -0.3]])
        rho = 0.3

        mean0 = np.array([[0.100438, -0.0846095],
                          [-0.00859508, -0.033298],
                          [0.0764211, -0.0425793]])
        var0 = 1/3*(np.array([[0.111863, -0.0537709],
                              [-0.0537709, 0.114663]]) +
                    np.array([[0.0534788, -0.0132181],
                              [-0.0132181, 0.0700114]]) +
                    np.array([[0.0901964, -0.0328068],
                              [-0.0328068, 0.0839981]]))

        mean, var = priors.f_in_jointly_gauss_bernoulli(A, B, rho)

        self.assertTrue(np.allclose(mean, mean0),
                        msg='mean wrong')
        self.assertTrue(np.allclose(var, var0),
                        msg='variance wrong')


class GaussBernoulliTests(unittest.TestCase):
    def test_f_in_assuming_diagA(self):
        '''
        Check the implementation of the prior for N distributions, ASSUMING THAT
        A IS DIAGONAL!
        '''
        # generate some dummy data
        B = np.array([[-1.47833, -2.25989, -0.787138],
                      [-0.0931732, 0.908117, -0.136544]])
        N, r = B.shape
        A = 3 * rnd.rand(r, r)
        rho = 1/2.72

        # calculated in Mathematica
        means0 = np.array([[-1.23716, -2.16209, -0.551793],
                           [-0.0590123, 0.655707, -0.0866396]])
        var0 = np.diag([0.885304, 1.02787, 0.734857])

        means, var = priors.f_in_gauss_bernoulli_assuming_diagA(A, B, rho)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')


class SBMTests(unittest.TestCase):
    def test_f_in_multiple_differentA_avgCovFalse(self):
        '''
        Check the implementation of the prior for N distributions with A having
        shape (n, r, r) when it should return N covariance matrices
        '''
        # generate some dummy data
        N = 4
        r = 2
        A = 3*rnd.rand(N, r, r)
        B = rnd.randn(N, r)

        x0 = np.array([1, 0])
        x1 = np.array([0, 1])

        # First, compute explicitly the means...
        means0 = np.array([x0*self.__W(x0, A[n], B[n]) +
                           x1*self.__W(x1, A[n], B[n])
                           for n in range(N)])
        # and the four covariances for the N distributions
        var0 = np.array([
            np.outer(x0, x0)*self.__W(x0, A[n], B[n])
            + np.outer(x1, x1)*self.__W(x1, A[n], B[n])
            - np.outer(means0[n], means0[n]) for n in range(N)])

        means, var, logZ = priors.f_in_sbm(A, B, avgCov=False)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')

    def test_f_in_multiple_differentA(self):
        '''
        Check the implementation of the prior for N distributions with A having
        shape (n, r, r)
        '''
        # generate some dummy data
        N = 4
        r = 2
        A = 3*rnd.rand(N, r, r)
        B = rnd.randn(N, r)

        x0 = np.array([1, 0])
        x1 = np.array([0, 1])

        # First, compute explicitly the means...
        means0 = np.array([x0*self.__W(x0, A[n], B[n]) +
                           x1*self.__W(x1, A[n], B[n])
                           for n in range(N)])
        # and the four covariances for the N distributions
        var0 = np.array([
            np.outer(x0, x0)*self.__W(x0, A[n], B[n])
            + np.outer(x1, x1)*self.__W(x1, A[n], B[n])
            - np.outer(means0[n], means0[n]) for n in range(N)])
        var0 = np.mean(var0, axis=0)

        means, var, logZ = priors.f_in_sbm(A, B)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')

    def test_f_in_multiple_sameA(self):
        '''
        Check the implementation of the prior for N distributions, where each
        distribution shares the same A matrix.
        '''
        # generate some dummy data
        N = 4
        r = 2
        A = 3*rnd.rand(r, r)
        B = rnd.randn(N, r)

        x0 = np.array([1, 0])
        x1 = np.array([0, 1])

        # First, compute explicitly the means...
        means0 = np.array([x0*self.__W(x0, A, B[n]) +
                           x1*self.__W(x1, A, B[n])
                           for n in range(N)])
        # and the four covariances for the N distributions
        var0 = np.array([
            np.outer(x0, x0)*self.__W(x0, A, B[n])
            + np.outer(x1, x1)*self.__W(x1, A, B[n])
            - np.outer(means0[n], means0[n]) for n in range(N)])
        var0 = np.mean(var0, axis=0)

        # and the log(Z)
        # well aware of the redundancy in the following,
        # just want to be as explicit as possible
        Z0 = np.zeros(N)
        for n in range(N):
            for i in range(r):
                xi = np.eye(r)[i]
                Z0[n] += 1 / r * np.exp(B[n] @ xi - xi.T @ A @ xi / 2)
        logZ0 = np.log(Z0)

        means, var, logZ = priors.f_in_sbm(A, B)

        self.assertTrue(np.allclose(means, means0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')
        self.assertTrue(np.allclose(logZ, logZ0), msg='log(Z) wrong')

    def test_f_in_single(self):
        '''
        Check the implementation of the prior for a single distribution.
        '''
        # generate some dummy data
        r = 2
        A = 3*rnd.rand(r, r)
        B = rnd.randn(r)

        x0 = np.array([1, 0])
        x1 = np.array([0, 1])

        # First, compute explicitly the means...
        mean0 = x0*self.__W(x0, A, B) + x1*self.__W(x1, A, B)
        # and the four covariances for the N distributions
        var0 = (np.outer(x0, x0)*self.__W(x0, A, B)
                + np.outer(x1, x1)*self.__W(x1, A, B)
                - np.outer(mean0, mean0))

        mean, var, logZ = priors.f_in_sbm(A, np.array([B]))

        self.assertTrue(np.allclose(mean, mean0), msg='mean wrong')
        self.assertTrue(np.allclose(var, var0), msg='cov wrong')

    def test_W_normalisation(self):
        N = 4
        r = 2
        A = 3*rnd.rand(r, r)
        B = rnd.randn(N, r)

        x0 = np.array([1, 0])
        x1 = np.array([0, 1])

        # compute Z for one of the distributions, here the one with idx=2
        Z = self.__W(x0, A, B[2]) + self.__W(x1, A, B[2])
        self.assertAlmostEqual(1, Z)

    def __W(self, x, A, B):
        """
        Distribution for a single mean B and single covariance matrix A.

        Parameters:
        -----------
        x : (r,)
            1-of-r representation of a community,
            i.e. x = (1, 0, ...), (0, 1, 0, ...) etc.
        A : (r, r)
        B : (r,)

        Returns
        -------
        scalar
        """
        r = x.shape[0]
        # well aware of the redundancy in the following,
        # just want to be as explicit as possible
        Z = 0
        for i in range(r):
            xi = np.eye(r)[i]
            Z += 1 / r * np.exp(B @ xi - xi.T @ A @ xi / 2)
        return 1 / Z * 1 / r * np.exp(B @ x - x.T @ A @ x / 2)


if __name__ == '__main__':
    unittest.main()
