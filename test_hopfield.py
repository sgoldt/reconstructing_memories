#!/usr/bin/env python3
"""
Tests for the Hopfield library.

Author: Sebastian Goldt <sebastian.goldt@ipht.fr>

Version: 0.1 :: 20180310
"""

import numpy as np

import unittest

import hopfield as hf


class HopfieldTests(unittest.TestCase):
    def test_binarise(self):
        x = np.array([[0.58571881, -0.14041748, -1.54543365],
                      [-1.63513868, 0, 1.59794409],
                      [0.33084115,  -0, -1.93456832]])
        expected = np.array([[1, -1, -1],
                             [-1, -1,  1],
                             [1,  -1, -1]]).astype(x.dtype)

        self.assertTrue(np.allclose(expected, hf.binarise(x)))

    def test_Delta(self):
        # numerical values provided by Mathematica
        nu = 0.2
        tau = 0
        delta_expected = 0.04888123763
        self.assertAlmostEqual(delta_expected, hf.get_Delta(nu, tau))

        nu = 0.3
        tau = 0.1
        delta_expected = 0.124810021
        self.assertAlmostEqual(delta_expected, hf.get_Delta(nu, tau))

    def test_S(self):
        # numerical values provided by Mathematica
        J = np.array([[0.3755974455, -1.941129800, -1.045928087],
                      [1.105057779, 0.7690869463, -0.7530072663],
                      [-0.2398215289, 0.3672519117, -0.7212020803]])
        nu = 0.2
        tau = 0
        S_expected = np.array([[9.389936137, -3.989422804, -3.989422804],
                               [27.62644447, 19.22717366, -3.989422804],
                               [-3.989422804, 9.181297792, -3.989422804]])
        self.assertTrue(np.allclose(S_expected, hf.get_S(J, nu, tau),
                                    atol=1e-7))

        nu = 0.3
        tau = 0.1
        S_expected = np.array([[5.284416061, -1.994967594, -1.994967594],
                               [13.38953087, 9.656521626, -1.994967594],
                               [-1.994967594, 5.191687908, -1.994967594]])
        self.assertTrue(np.allclose(S_expected, hf.get_S(J, nu, tau),
                                    atol=1e-7))


if __name__ == '__main__':
    unittest.main()
