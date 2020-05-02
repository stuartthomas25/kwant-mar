# Copyright 2017, 2018 Christoph Groth (CEA).
#
# This file is part of Vquad.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import bisect

import numpy as np
from scipy.linalg import norm

from . import tables as tbls

eps = np.spacing(1)

# If the relative difference between two consecutive approximations is
# lower than this value, the error estimate is considered reliable.
# See section 6.2 of Pedro Gonnet's thesis.
hint = 0.1

# Smallest acceptable relative difference of points in a rule.  This was chosen
# such that no artifacts are apparent in plots of (i, log(a_i)), where a_i is
# the sequence of estimates of the integral value of an interval and all its
# ancestors..
min_sep = 16 * eps

min_level = 1
max_level = 4

ndiv_max = 20

_sqrt_one_half = np.sqrt(0.5)


def _eval_legendre(c, x):
    """Evaluate _orthonormal_ Legendre polynomial.

    This uses the three-term recurrence relation from page 63 of Perdo
    Gonnet's thesis.
    """
    if len(c) <= 1:
        c0 = c[0]
        c1 = 0
    else:
        n = len(c)
        c0 = c[-2]              # = c[k + 0]
        c1 = c[-1]              # = c[k + 1]
        for k in range(len(c) - 2, 0, -1):
            a = (2*k + 3) / (k + 1)**2
            tmp = c0
            c0 = c[k - 1] - c1 * np.sqrt(a * k**2 / (2*k - 1))
            c1 = tmp + c1 * x * np.sqrt(a * (2*k + 1))

    return np.sqrt(1/2) * c0 + np.sqrt(3/2) * c1 * x


def _calc_coeffs(vals, level):
    nans = np.flatnonzero(~np.isfinite(vals))
    if nans.size:
        # Replace vals by a copy and zero-out non-finite elements.
        vals = vals.copy()
        vals[nans] = 0
        # Prepare things for the loop further down.
        b = tbls.newton_coeffs[level].copy()
        m = len(b) - 2              # = len(tbls.nodes[level]) - 1
    coeffs = tbls.inv_Vs[level] @ vals

    # This is a variant of Algorithm 7 from the thesis of Pedro Gonnet where no
    # linear system has to be solved explicitly.  Instead, Algorithm 5 is used.
    for i in nans:
        b[m + 1] /= tbls.alpha[m]
        x = tbls.nodes[level][i]
        b[m] = (b[m] + x * b[m + 1]) / tbls.alpha[m - 1]
        for j in range(m - 1, 0, -1):
            b[j] = ((b[j] + x * b[j + 1] - tbls.gamma[j + 1] * b[j + 2])
                    / tbls.alpha[j - 1])
        b = b[1:]

        coeffs[:m] -= coeffs[m] / b[m] * b[:m]
        coeffs[m] = 0
        m -= 1

    return coeffs


class DivergentIntegralError(ValueError):
    def __init__(self, msg, igral, err):
        self.igral = igral
        self.err = err
        super().__init__(msg)


class _Terminator:
    __slots__ = ['prev', 'next']


class _Interval:
    __slots__ = ['a', 'b', 'coeffs', 'vals', 'igral', 'err', 'level', 'depth',
                 'ndiv', 'c00', 'unreliable_err', 'prev', 'next']

    def __init__(self, a, b, level, depth):
        self.a = a
        self.b = b
        self.level = level
        self.depth = depth

    def map(self, points):
        a = self.a
        b = self.b
        return (a + b) / 2 + (b - a) / 2 * points

    def interpolate(self, vals, coeffs_old=None):
        self.vals = vals
        self.coeffs = coeffs = _calc_coeffs(self.vals, self.level)
        if self.level == min_level:
            self.c00 = coeffs[0]
        if coeffs_old is None:
            coeffs_diff = norm(coeffs)
        else:
            coeffs_diff = np.zeros(max(len(coeffs_old), len(coeffs)))
            coeffs_diff[:len(coeffs_old)] = coeffs_old
            coeffs_diff[:len(coeffs)] -= coeffs
            coeffs_diff = norm(coeffs_diff)
        w = self.b - self.a
        self.igral = w * coeffs[0] * _sqrt_one_half
        self.err = w * coeffs_diff
        self.unreliable_err = coeffs_diff > hint * norm(coeffs)

    def __lt__(self, other):
        return self.err < other.err

    def __call__(self, x):
        a = self.a
        b = self.b
        x = (2 * x - (a + b)) / (b - a)
        return _eval_legendre(self.coeffs, x)

    def split(self):
        """Split this interval in the center into two children.

        This is a coroutine that initially yields an array of x values
        of points to be evaluated.  Once the corresponding values have
        been sent back a tuple containing the child intervals is
        yielded and execution ends.
        """
        m = (self.a + self.b) / 2
        f_center = self.vals[(len(self.vals) - 1) // 2]

        depth = self.depth + 1
        children = [_Interval(self.a, m, min_level, depth),
                    _Interval(m, self.b, min_level, depth)]
        points = tbls.nodes[min_level][1:-1]
        points = np.concatenate([child.map(points) for child in children])
        valss = np.empty((2, tbls.sizes[min_level]))
        valss[:, 0] = self.vals[0], f_center
        valss[:, -1] = f_center, self.vals[-1]
        valss[:, 1:-1] = (yield points).reshape((2, -1))

        for child, vals, T in zip(children, valss, tbls.Ts):
            child.interpolate(vals, T[:, :self.coeffs.shape[0]] @ self.coeffs)
            child.ndiv = (self.ndiv
                         + (self.c00 and child.c00 / self.c00 > 2))
            if child.ndiv > ndiv_max and 2*child.ndiv > child.depth:
                msg = ('Possibly divergent integral in the interval '
                       '[{}, {}]! (h={})')
                raise DivergentIntegralError(
                    msg.format(child.a, child.b, child.b - child.a),
                    child.igral * np.inf, None)
        yield children

    def refine(self):
        """Increase degree of interval.

        This is a coroutine that initially yields an array of x values
        of points to be evaluated.  Once the corresponding values have
        been sent back, a bool is yielded and execution ends.

        It is "true" if further refinements/splits of the interval seem
        promising, and "false" otherwise.  This is the case when
        neigboring points can be resolved only barely by floating point
        numbers, or when the estimated relative error is already at the
        limit of numerical accuracy and cannot be reduced further.
        """
        self.level += 1
        points = self.map(tbls.nodes[self.level])
        vals = np.empty(points.shape)
        vals[0::2] = self.vals
        vals[1::2] = (yield points[1::2])
        self.interpolate(vals, self.coeffs)

        yield (points[1] - points[0] > points[0] * min_sep
               and points[-1] - points[-2] > points[-2] * min_sep
               and self.err > (abs(self.igral)
                               * eps * tbls.V_cond_nums[self.level]))


class Vquad:
    """Evaluate an integral using adaptive quadrature.

    The algorithm uses Clenshaw-Curtis quadrature rules of increasing
    degree in each interval.  The error estimate is
    sqrt(integrate((f0(x) - f1(x))**2)), where f0 and f1 are two
    successive interpolations of the integrand.  To fall below the
    desired total error, intervals are worked on ranked by their own
    absolute error: either the degree of the rule is increased or the
    interval is split if either the function does not appear to be
    smooth or a rule of maximum degree has been reached.

    Reference: "Increasing the Reliability of Adaptive Quadrature
        Using Explicit Interpolants", P. Gonnet, ACM Transactions on
        Mathematical Software, 37 (3), art. no. 26, 2008.
    """

    def __init__(self, f, a, b, level=max_level):
        ival = _Interval(a, b, level - 1, 1)
        ival.interpolate(f(ival.map(tbls.nodes[level - 1])))
        ival.c00 = 0.0          # Will go away.
        ival.ndiv = 0

        self.ivals = [ival]     # Active intervals
        self.f = f
        self.igral_excess = 0
        self.err_excess = 0

        # Initialize linked list.
        ival.prev = self.begin = _Terminator()
        self.begin.next = ival
        ival.next = self.end = _Terminator()
        self.end.prev = ival

        # Refine up to requested level.  This calculates a proper error
        # estimate.
        self.improve()

    def improve(self):
        ival = self.ivals[-1]

        if ival.level == max_level:
            split = True
        else:
            refine = ival.refine()
            if not refine.send(self.f(next(refine))):
                # Remove the interval but remember the excess integral and
                # error.
                self.err_excess += ival.err
                self.igral_excess += ival.igral
                self.ivals.pop()
                return
            split = ival.unreliable_err

        if split:
            # Replace current interval by its children.
            self.ivals.pop()
            split = ival.split()
            child0, child1 = split.send(self.f(next(split)))
            bisect.insort(self.ivals, child0)
            bisect.insort(self.ivals, child1)

            # Maintain linked list.
            ival.prev.next = child0
            ival.next.prev = child1
            child0.prev = ival.prev
            child0.next = child1
            child1.prev = child0
            child1.next = ival.next
        else:
            # The error estimate of the current interval has changed.
            bisect.insort(self.ivals, self.ivals.pop())

    def totals(self):
        igral = self.igral_excess
        err = self.err_excess
        for ival in self.ivals:
            igral += ival.igral
            err += ival.err
        return igral, err

    def improve_until(self, rtol=0, atol=0):
        if rtol < 0 or atol < 0:
            raise ValueError("Tolerances must be positive.")
        if rtol == 0 and atol == 0:
            raise ValueError("Either rtol or atol must be nonzero.")

        while True:
            igral, err = self.totals()
            tol = max(atol, abs(igral) * rtol)
            if (err == 0 or err < tol
                or self.err_excess > tol > err - self.err_excess
                or not self.ivals):
                return igral, err
            self.improve()

    def __call__(self, xs):
        xs = np.asarray(xs)
        shape = xs.shape
        if xs.size == 0:
            return np.empty(shape)
        xs = xs.flatten()

        # Sort xs, but remember inverse permutation.
        perm = np.argsort(xs)
        xs = xs[perm]
        inv_perm = np.empty(len(perm), int)
        inv_perm[perm] = np.arange(len(perm))

        # Evaluate points interval by interval.
        results = []
        ival = self.begin.next
        end = self.end
        if xs[0] < ival.a or xs[-1] > end.prev.b:
            raise ValueError("Point lies outside of integration interval.")
        i = 0
        while ival is not end:
            j = bisect.bisect(xs, ival.b, i)
            if j != i:
                results.append(ival(xs[i:j]))
                i = j
            ival = ival.next

        return np.concatenate(results)[inv_perm].reshape(shape)


def vquad(f, a, b, rtol=0, atol=0):
    igrator = Vquad(f, a, b)
    return igrator.improve_until(rtol, atol)
