# Copyright 2017 Christoph Groth (CEA).
#
# This file is part of Vquad.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from fractions import Fraction as Frac
from itertools import combinations
import numpy as np
from numpy.testing import assert_allclose

from .. import tables


def test_legendre():
    legs = tables.legendre(11)
    comparisons = [(legs[0], [1], 1),
                    (legs[1], [0, 1], 1),
                    (legs[10], [-63, 0, 3465, 0, -30030, 0,
                                90090, 0, -109395, 0, 46189], 256)]
    for a, b, div in comparisons:
        for c, d in zip(a, b):
            assert c * div == d


def test_scalar_product(n=33):
    legs = tables.legendre(n)
    selection = [0, 5, 7, n-1]
    for i in selection:
        for j in selection:
            assert (tables.scalar_product(legs[i], legs[j])
                    == ((i == j) and Frac(2, 2*i + 1)))


def simple_newton(n):
    """Slower than 'newton()' and prone to numerical error."""
    nodes = -np.cos(np.arange(n) / (n-1) * np.pi)
    return [sum(np.prod(-np.asarray(sel))
                for sel in combinations(nodes, n - d))
            for d in range(n + 1)]


def test_newton():
    assert_allclose(tables.newton(9), simple_newton(9), atol=1e-15)


def test_newton_legendre(level=1):
    legs = [np.array(leg, float)
            for leg in tables.legendre(tables.sizes[level] + 1)]
    result = np.zeros(len(legs[-1]))
    for factor, leg in zip(tables.newton_coeffs[level], legs):
        factor *= np.sqrt((2*len(leg) - 1) / 2)
        result[:len(leg)] += factor * leg
    assert_allclose(result, tables.newton(tables.sizes[level]), rtol=1e-15)
