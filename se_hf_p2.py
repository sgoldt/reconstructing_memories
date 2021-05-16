#!/usr/bin/env python3

"""
Explicit state-evolution for the rectified Hopfield model with P=2.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180404
"""

import numpy as np
import numpy.random as rnd
from scipy import integrate
import scipy.linalg as la

t_max = 100

DESCRIPTION = '''
Explicit state-evolution for the rectified Hopfield model with P=2.
'''

xis = np.array([[1, 1],
                [1, -1],
                [-1, 1],
                [-1, -1]])


def f_in(a2, b1, b2):
    exp1 = np.exp(2*(a2 + b1))
    exp2 = np.exp(2*(a2 + b2))
    exp3 = np.exp(2*(b1 + b2))
    f1 = -1 + exp1 - exp2 + exp3
    f2 = -1 - exp1 + exp2 + exp3
    Z = 1 + exp1 + exp2 + exp3
    return np.array([f1/Z, f2/Z])


def gauss(w1, w2):
    return 1 / (2 * np.pi) * np.exp(-w1**2/2 - w2**2/2)


def integrand_m1(w1, w2, m1, m2, m3, Delta):
    X = (np.array([[m1, m2], [m2, m3]]) / Delta)
    noise = la.sqrtm(X) @ np.array([w1, w2])
    B = X @ xis.T + noise[:, None]

    return gauss(w1, w2) / 4 * (f_in(X[0, 1], *B[:, 0])
                                + f_in(X[0, 1], *B[:, 1])
                                - f_in(X[0, 1], *B[:, 2])
                                - f_in(X[0, 1], *B[:, 3]))[0]


def integrand_m2(w1, w2, m1, m2, m3, Delta):
    X = (np.array([[m1, m2], [m2, m3]]) / Delta)
    noise = la.sqrtm(X) @ np.array([w1, w2])
    B = X @ xis.T + noise[:, None]

    return gauss(w1, w2) / 4 * (f_in(X[0, 1], *B[:, 0])
                                - f_in(X[0, 1], *B[:, 1])
                                + f_in(X[0, 1], *B[:, 2])
                                - f_in(X[0, 1], *B[:, 3]))[0]


def integrand_m3(w1, w2, m1, m2, m3, Delta):
    X = (np.array([[m1, m2], [m2, m3]]) / Delta)
    noise = la.sqrtm(X) @ np.array([w1, w2])
    B = X @ xis.T + noise[:, None]

    return gauss(w1, w2) / 4 * (f_in(X[0, 1], *B[:, 0])
                                - f_in(X[0, 1], *B[:, 1])
                                + f_in(X[0, 1], *B[:, 2])
                                - f_in(X[0, 1], *B[:, 3]))[1]


def main():
    rnd.seed(0)

    for Delta in np.linspace(.1, 1.2, 12):
        # draw a random pos. sem-def matrix
        m1, m3 = rnd.rand(2)  # diagonal elements
        m2 = np.sqrt(m1*m3) * rnd.rand()

        print('%d, %g, %g, %g, %g' % (0, Delta, m1, m2, m3))
        for t in range(1, t_max):
            m1_next = integrate.dblquad(integrand_m1,
                                        -5, 5, lambda x: -5, lambda x: 5,
                                        args=(m1, m2, m3, Delta))
            m2_next = integrate.dblquad(integrand_m2,
                                        -5, 5, lambda x: -5, lambda x: 5,
                                        args=(m1, m2, m3, Delta))
            m3_next = integrate.dblquad(integrand_m3,
                                        -5, 5, lambda x: -5, lambda x: 5,
                                        args=(m1, m2, m3, Delta))
            diff = (np.abs(m1 - m1_next[0])
                    + np.abs(m2 - m2_next[0])
                    + np.abs(m3 - m3_next[0]))

            m1 = m1_next[0]
            m2 = m2_next[0]
            m3 = m3_next[0]

            print('%d, %g, %g, %g, %g' % (t, Delta, m1, m2, m3))

            if diff < 1e-6:
                print()
                break

    print("Bye-bye!")


if __name__ == "__main__":
    main()
