# Copyright 2017, 2018 Christoph Groth (CEA).
#
# This file is part of Vquad.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

from math import exp, log, log10
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

from .. import core


def f1(x):
    return np.exp(x)

def f2(x):
    return x >= 0.3

def f3(x):
    return np.sqrt(x)

def f4(x):
    return 23/25 * np.cosh(x) - np.cos(x)

def f5(x):
    xx = x*x
    return 1 / ( xx * (xx + 1) + 0.9)

def f6(x):
    return x * np.sqrt(x)

def f7(x):
    return 1 / np.sqrt(x)

def f8(x):
    xx = x*x
    return 1 / (1 + xx*xx)

def f9(x):
    return 2 / (2 + np.sin(10 * np.pi * x))

def f10(x):
    return 1 / (1 + x)

def f11(x):
    return 1 / (1 + np.exp(x))

def f12(x):
    return x / (np.exp(x) - 1)

def f13(x):
    return np.sin(100 * np.pi * x) / (np.pi * x)

def f14(x):
    return np.sqrt(50) * np.exp(-50 * np.pi * x * x)

def f15(x):
    return 25 * np.exp(-25 * x)

def f16(x):
    return 50 / np.pi * (2500 * x*x + 1)

def f17(x):
    t = 50 * np.pi * x
    t = np.sin(t) / t
    return 50 * t * t

def f18(x):
    return np.cos( np.cos(x) + 3 * np.sin(x) + 2 * np.cos(2*x)
                   + 3 * np.sin(2*x) + 3 * np.cos(3*x) )

def f19(x):
    return np.log(x)

def f20(x):
    return 1 / (1.005 + x*x)

def f21(x):
    return sum(1 / np.cosh(20**i * (x - 2*i/10)) for i in range(1, 4))

def f22(x):
    return 4 * np.pi**2 * x * np.sin(20 * np.pi * x) * np.cos(2 * np.pi * x)

def f23(x):
    t = 230 * x - 30
    return 1 / (1 + t*t)

def f24(x):
    return np.floor(np.exp(x))

def f25(x):
    return ((x + 1) * (x < 1)
            + (3 - x) * ((1 <= x) & (x <= 3))
            + 2 * (x > 3))

battery = [
    (f1, (0, 1), 1.7182818284590452354),
    (f2, (0, 1), 0.7),
    (f3, (0, 1), 2/3),
    (f4, (-1, 1), 0.4794282266888016674),
    (f5, (-1, 1), 1.5822329637296729331),
    (f6, (0, 1), 0.4),
    (f7, (0, 1), 2),
    (f8, (0, 1), 0.86697298733991103757),
    (f9, (0, 1), 1.1547005383792515290),
    (f10, (0, 1), 0.69314718055994530942),
    (f11, (0, 1), 0.3798854930417224753),
    (f12, (0, 1), 0.77750463411224827640),
    (f13, (0, 1), 0.49898680869304550249),
    (f14, (0, 10), 0.5),
    (f15, (0, 10), 1),
    (f16, (0, 10), 0.13263071079267703209e+08),
    (f17, (0, 1), 0.49898680869304550249),
    (f18, (0, np.pi), 0.83867634269442961454),
    (f19, (0, 1), -1),
    (f20, (-1, 1), 1.5643964440690497731),
    (f21, (0, 1), 0.16349494301863722618),
    (f22, (0, 1), -0.63466518254339257343),
    (f23, (0, 1), 0.013492485649467772692),
    (f24, (0, 3), 17.664383539246514971),
    (f25, (0, 5), 7.5),
]

def test_battery():
    """Test and gather statistics on a battery of difficult integrands

    The battery is used as listed on page 168 of Pedro Gonnet's thesis.

    Reference:
    W. Gander and W. Gautschi. Adaptive quadrature — revisited.
    Technical Report 306, Department of Computer Science, ETH
    Zurich, Switzerland, 1998.
    """
    def ff(x):
        nonlocal neval
        neval += len(x)
        return f(x)

    old_settings = np.seterr(all='ignore')

    rtols = [1e-3, 1e-6, 1e-9, 1e-12]
    sum_log_neval = 0
    sum_extra_digits = 0
    sum_nonrep_digits = 0
    for f, (a, b), exact in battery:
        for rtol in rtols:
            neval = 0
            igral, err = core.vquad(ff, a, b, rtol)
            sum_log_neval += log(neval)
            extra_digits = min(16, log10(rtol / abs((igral - exact) / exact)))
            sum_extra_digits += extra_digits
            nonrep_digits = min(16, log10(err / abs(igral - exact)))
            sum_nonrep_digits += nonrep_digits
            if not (f is f21 and rtol > 1e-11):
                assert extra_digits > 0
                assert nonrep_digits > 0

    n = len(rtols) * len(battery)
    print("geom. mean of number of evaluations:",
          round(exp(sum_log_neval / n), 2), "(cquad: 237.92)")
    print("mean non-requested significant digits:",
          round(sum_extra_digits / n, 3),
          "(cquad: 5.989)")
    print("mean non-reported significant digits:",
          round(sum_nonrep_digits / n, 3),
          "(cquad: 4.056)")

    np.seterr(**old_settings)


def f_one_with_nan(x):
    x = np.asarray(x)
    result = np.ones(x.shape)
    result[x == 0] = np.inf
    return result


def test_one_with_nan():
    rtol = 1e-12
    exact = 2
    igral, err = core.vquad(f_one_with_nan, -1, 1, rtol)
    assert_allclose(igral, exact, rtol)
    assert err / exact < rtol


def test_coeffs(level=3):
    vals = np.abs(core.tbls.nodes[level])
    vals[1::2] = np.nan
    c_downdated = core._calc_coeffs(vals, level)

    level -= 1
    vals = np.abs(core.tbls.nodes[level])
    c = core._calc_coeffs(vals, level)
    assert_allclose(c_downdated[:len(c)], c, rtol=0, atol=1e-12)

    vals_from_c = core._eval_legendre(c, core.tbls.nodes[level])
    assert_allclose(vals_from_c, vals, rtol=0, atol=1e-15)


def test_interpolation():
    vquad = core.Vquad(f24, 0, 3)
    vquad.improve_until(1e-6)

    rng = np.random.RandomState(123)
    x = np.linspace(0, 3, 100)
    rng.shuffle(x)

    for x in [x, 1.23, [[2, 0], [1, 2]], [], [[]]]:
        assert_allclose(vquad(x), f24(x))

    for x in [-1e-100, 3.0001, 1e100, [1, 2, -1e50]]:
        raises(ValueError, vquad, x)


def test_divergence_detection(n=200):
    def f(x):
        x = x.reshape((-1, 1))
        return (c * abs(x - λ) ** α).sum(1)

    def F(x):
        x = x.reshape((-1, 1))
        return (c * (x - λ) * abs(x - λ) ** α / (α + 1)).sum(1)

    old_settings = np.seterr(all='ignore')

    rng = np.random.RandomState(123)
    false_negatives = 0
    false_positives = 0

    shape = (1, 7)
    num_divergent = 0
    for i in range(n):
        α = rng.uniform(-0.1, -1.1, shape)
        λ = rng.uniform(0, 1, shape)
        c = np.exp(rng.uniform(-8, 0, shape)) * rng.choice((-1, 1), shape)
        # The above is chosen such that the integral is divergent about 50% of
        # the time.
        divergent = (α <= -1).any()
        num_divergent += divergent
        try:
            vquad = core.Vquad(f, 0, 1)
            igral, err = vquad.improve_until(1e-3)
        except core.DivergentIntegralError:
            false_negatives += not divergent
        else:
            if divergent:
                false_positives += 1

    print("false negatives (non-divergent but failure):",
          false_negatives, "/", n - num_divergent)
    print("false positives (divergent but success):",
          false_positives, "/", num_divergent)

    # Hopefully we can improve on the following!
    assert false_negatives < 0.3 * (n - num_divergent)
    assert false_positives < 0.3 * num_divergent

    np.seterr(**old_settings)
