# Copyright 2017 Christoph Groth (CEA).
#
# This file is part of Vquad.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from fractions import Fraction as Frac
from collections import defaultdict
import numpy as np
from scipy.linalg import norm, inv

__all__ = ['sizes', 'nodes', 'newton_coeffs', 'inv_Vs', 'V_cond_nums', 'Ts',
           'alpha', 'gamma']

def legendre(n):
    """Return the first n Legendre polynomials.

    The polynomials have *standard* normalization, i.e.
    int_{-1}^1 dx L_n(x) L_m(x) = delta(m, n) * 2 / (2 * n + 1).

    The return value is a list of list of fraction.Fraction instances.
    """
    result = [[Frac(1)], [Frac(0), Frac(1)]]
    if n <= 2:
        return result[:n]
    for i in range(2, n):
        # Use Bonnet's recursion formula.
        new = (i + 1) * [Frac(0)]
        new[1:] = (r * (2*i - 1) for r in result[-1])
        new[:-2] = (n - r * (i - 1) for n, r in zip(new[:-2], result[-2]))
        new[:] = (n / i for n in new)
        result.append(new)
    return result


def newton(n):
    """Compute the monomial coefficients of the Newton polynomial over the
    nodes of the n-point Clenshaw-Curtis quadrature rule.
    """
    # The nodes of the Clenshaw-Curtis rule are x_i = -cos(i * Pi / (n-1)).
    # Here, we calculate the coefficients c_i such that sum_i c_i * x^i
    # = prod_i (x - x_i).  The coefficients are thus sums of products of
    # cosines.
    #
    # This routine uses the relation
    #   cos(a) cos(b) = (cos(a + b) + cos(a - b)) / 2
    # to efficiently calculate the coefficients.
    #
    # The dictionary 'terms' descibes the terms that make up the
    # monomial coefficients.  Each item ((d, a), m) corresponds to a
    # term m * cos(a * Pi / n) to be added to prefactor of the
    # monomial x^(n-d).

    mod = 2 * (n-1)
    terms = defaultdict(int)
    terms[0, 0] += 1

    for i in range(n):
        newterms = []
        for (d, a), m in terms.items():
            for b in [i, -i]:
                # In order to reduce the number of terms, cosine
                # arguments are mapped back to the inteval [0, pi/2).
                arg = (a + b) % mod
                if arg > n-1:
                    arg = mod - arg
                if arg >= n // 2:
                    if n % 2 and arg == n // 2:
                        # Zero term: ignore
                        continue
                    newterms.append((d + 1, n - 1 - arg, -m))
                else:
                    newterms.append((d + 1, arg, m))
        for d, s, m in newterms:
            terms[d, s] += m

    c = (n + 1) * [0]
    for (d, a), m in terms.items():
        if m and a != 0:
            raise ValueError("Newton polynomial cannot be represented exactly.")
        c[n - d] += m
        # The check could be removed and the above line replaced by
        # the following, but then the result would be no longer exact.
        # c[n - d] += m * np.cos(a * np.pi / (n - 1))

    cf = np.array(c, float)
    assert all(int(cfe) == ce for cfe, ce in zip(cf, c)), 'Precision loss'

    cf /= 2.**np.arange(n, -1, -1)
    return cf


def scalar_product(a, b):
    """Compute the polynomial scalar product int_-1^1 dx a(x) b(x).

    The args must be sequences of polynomial coefficients.  This
    function is careful to use the input data type for calculations.
    """
    la = len(a)
    lc = len(b) + la + 1

    # Compute the even coefficients of the product of a and b.
    c = lc * [a[0].__class__()]
    for i, bi in enumerate(b):
        if bi == 0:
            continue
        for j in range(i % 2, la, 2):
            c[i + j] += a[j] * bi

    # Calculate the definite integral from -1 to 1.
    return 2 * sum(c[i] / (i + 1) for i in range(0, lc, 2))


def newton_legendre(sizes):
    """Calculate the decompositions of Newton polynomials (over the nodes
    of the n-point Clenshaw-Curtis quadrature rule) in terms of
    Legandre polynomials.

    The parameter 'sizes' is a sequence of numers of points of the
    quadrature rule.  The return value is a corresponding sequence of
    normalized Legendre polynomial coefficients.
    """
    legs = legendre(max(sizes) + 1)
    result = []
    for n in sizes:
        poly = []
        a = list(map(Frac, newton(n)))
        for b in legs[:n + 1]:
            igral = scalar_product(a, b)

            # Normalize & store.  (The polynomials returned by
            # legendre() have standard normalization that is not
            # orthonormal.)
            poly.append(np.sqrt((2*len(b) - 1) / 2) * igral)

        result.append(np.array(poly))
    return result


def vandermonde(nodes):
    V = [np.ones(nodes.shape), nodes.copy()]
    for i in range(2, len(nodes)):
        V.append((2*i-1) / i * nodes * V[-1] - (i-1) / i * V[-2])
    for i in np.arange(len(nodes)):
        V[i] *= np.sqrt(i + 0.5)
    return np.array(V).T


def precalculate(n_levels=5):
    """Precalculate tables for adaptive quadrature based on the
       Clenshaw-Curtis rule and Legendre polynomials.

    The Clenshaw-Curtis rule with three points is the lowest rule that
    contains the center of the interval, hence we define it as level 0.
    """
    global sizes, nodes, newton_coeffs, inv_Vs, V_cond_nums, Ts, alpha, gamma

    # Points of the Clenshaw-Curtis rule.
    sizes = [2**(level + 1) + 1 for level in range(n_levels)]
    nodes = [-np.cos(np.pi / (n - 1) * np.arange(n)) for n in sizes]
    # Set central rule points precisely to zero.  This does not really
    # matter in practice, but is useful for tests.
    for l in nodes:
        l[len(l) // 2] = 0.0

    # Vandermonde-like matrices and their condition numbers
    V = list(map(vandermonde, nodes))
    inv_Vs = inv_Vs = list(map(inv, V))
    V_cond_nums = [norm(a, 2) * norm(b, 2) for a, b in zip(V, inv_Vs)]

    # Shift matrices
    Ts = [inv_Vs[-1] @ vandermonde((nodes[-1] + a) / 2) for a in [-1, 1]]

    # Newton polynomials
    newton_coeffs = newton_legendre(sizes)

    # Other downdate matrices
    k = np.arange(sizes[-1])
    alpha = np.sqrt((k+1)**2 / (2*k+1) / (2*k+3))
    gamma = np.concatenate([[0, 0],
                            np.sqrt(k[2:]**2 / (4*k[2:]**2-1))])


precalculate()
