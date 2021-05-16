#!/usr/bin/env python3

import numpy as np
import numpy.random as rnd

import itertools


def main():
    P = 3
    a = 0.7
    Delta = 0.2

    x0 = -1 + 2 * rnd.binomial(1, 0.5, (1, P))  # 1 trial, p=0.5, N x P array
    W = rnd.randn(P)
    A = a * np.eye(P)
    B = a / Delta * x0 + np.sqrt(a / Delta) * W

    f_in = np.zeros(P)
    Z = 0
    for p in itertools.product([-1, 1], repeat=P):
        x = np.array(p)
        f_in += np.exp(B @ x - 0.5 * x.T @ A @ x) * x
        Z += np.exp(B @ x - 0.5 * x.T @ A @ x)

    print(1 / Z * f_in)

    f_se = np.tanh(a / Delta * x0 + np.sqrt(a / Delta) * W)
    print(f_se)


if __name__ == "__main__":
    main()
