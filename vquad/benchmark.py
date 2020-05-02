# Copyright 2018 Christoph Groth (CEA).
#
# This file is part of Vquad.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution.

import math
import itertools
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.integrate

import matplotlib, matplotlib.cm
from matplotlib import pyplot

from . import core


def f63(x, α, λ):
    return np.abs(x - λ)**α

def F63_0_1(α, λ):
    def F(x):
        return (x - λ) * abs(x - λ)**α / (α + 1)
    return F(1) - F(0)


def f64(x, α, λ):
    return (x > λ) * np.exp(α * x)

def F64_0_1(α, λ):
    assert 0 <= λ <= 1
    if α == 0:
        return 1 - λ
    return (math.exp(α) - math.exp(α * λ)) / α


def f65(x, α, λ):
    return np.exp(-α * np.abs(x - λ))

def F65_0_1(α, λ):
    assert 0 <= λ <= 1
    if α == 0:
        return 1
    return (math.expm1(α*λ - α) + math.expm1(-α*λ)) / -α


# As shown in Pedro's thesis (Eq. (6.6) and Eq. (6.7)), the term 10^α is not
# squared in the denominator.  However, this is the correct version of the
# integrand that is consistent with the benchmark results that are shown in the
# thesis.
def f67(x, α, *λs):
    t = 10**α
    return t * sum(1 / ((x - λ)**2 + t**2) for λ in λs)

def F67_1_2(α, *λs):
    t = 10**α
    return sum(math.atan((2 - λ) / t) - math.atan((1 - λ) / t) for λ in λs)


# Equation (6.8) is the only one where the benchmark results do not agree with
# the ones listed in Pedro Gonnet's thesis.  So Eq. (6.8) seems to contain a
# typo.  I have modified the function by adding a small constant so that the
# definite integral is never zero.  Otherwise any required relative error
# cannot be obtained which leads to extremely long running times.
def f68(x, α, λ):
    β = 10**α / max(λ**2, (1 - λ)**2)
    return 2 + 2 * β * (x - λ) * np.cos(β * (x - λ)**2)

def F68_0_1(α, λ):
    β = 10**α / max(λ**2, (1 - λ)**2)
    return 2 + math.sin(β * ((1 - λ)**2)) - math.sin(β * λ**2)


families = [
    (r"|x - \lambda|^\alpha",
     (f63, 0, 1), F63_0_1, (-0.5, 0), (0, 1)),

    (r"(x - \lambda)\mathrm{e}^{\alpha x}",
     (f64, 0, 1), F64_0_1, (0, 1), (0, 1)),

    (r"\exp(-\alpha|x - \lambda|)",
     (f65, 0, 1), F65_0_1, (0, 4), (0, 1)),

    (r"10^\alpha/((x - \lambda)^2 + 10^{2\alpha})",
     (f67, 1, 2), F67_1_2, (-6, -3), (1, 2)),

    (r"\sum_i 10^\alpha/((x - \lambda_i)^2 + 10^{2\alpha})",
     (f67, 1, 2), F67_1_2, (-5, -3)) + ((1, 2),) * 4,

    (r"2 + 2\beta(x - \lambda) \cos(\beta (x - \lambda)^2),\quad"
     r"\beta = 10^\alpha / max(\lambda^2, (1 - \lambda)^2)",
     (f68, 0, 1), F68_0_1, (1.8, 2), (0, 1)),
]


def lk_job_vquad(seed, family, rtols):
    """Run a single integration for a sequence of requested relative
       tolerances.  It's as if a len(rtols) integration were started,
       but actually a single integrator is launched and the requested
       tolerance is improved iteratively.
    """
    def ff(x):
        nonlocal neval
        neval += len(x)
        return f(x, *args)

    _, (f, a, b), F, *bounds = family
    rng = np.random.RandomState(seed)
    args = [rng.uniform(*b) for b in bounds]

    relerrs = np.empty(len(rtols), float)
    nevals = np.empty(len(rtols), int)

    neval = 0
    exact = F(*args)
    vquad = core.Vquad(ff, a, b)
    for j, rtol in enumerate(rtols):
        igral, _ = vquad.improve_until(rtol)
        relerrs[j] = abs((igral - exact) / exact)
        nevals[j] = neval

    return nevals, relerrs


def lk_job_quadpack(seed, family, rtols):
    """Run a single integration for a sequence of requested relative
       tolerances.  It's as if a len(rtols) integration were started,
       but actually a single integrator is launched and the requested
       tolerance is improved iteratively.
    """
    def ff(x):
        nonlocal neval
        neval += 1
        return f(x, *args)

    _, (f, a, b), F, *bounds = family
    rng = np.random.RandomState(seed)
    args = [rng.uniform(*b) for b in bounds]

    relerrs = np.empty(len(rtols), float)
    nevals = np.empty(len(rtols), int)

    exact = F(*args)
    for j, rtol in enumerate(rtols):
        neval = 0
        igral, _ = scipy.integrate.quad(
                    ff, a, b, epsabs=0, epsrel=rtol)
        relerrs[j] = abs((igral - exact) / exact)
        nevals[j] = neval

    return nevals, relerrs


def lk_runner(map, n, job, *args):
    """Run job n times with different seed but otherwise same args.

    return two arrays of shape (n, -1) where the second dimension is
    determined by the args.
    """
    # map returns a sequence of (nevals, relerrs) pairs that is unzipped by the
    # zip(*...) construct.
    nevals, relerrs = zip(*map(job, range(n),
                               *(itertools.repeat(a) for a in args)))
    return np.stack(nevals), np.stack(relerrs)


def run(runner, job, n, rtols, thresholds, ):
    old_settings = np.seterr(all='ignore')

    executor = ProcessPoolExecutor()
    for family in families:
        nevals, relerrs = runner(executor.map, n, job, family, rtols)
        nevals = np.mean(nevals, axis=0)
        relerrs.sort(axis=0)
        successratios = [np.searchsorted(*args) / n
                         for args in zip(relerrs.T, rtols)]
        quantiles = {t: relerrs[int(t * n)] for t in thresholds}
        yield nevals, quantiles, successratios

    np.seterr(**old_settings)


def display(datasets, dataset_names, familynames, cols, colnames):
    matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(
        'color', matplotlib.cm.tab20.colors)

    fig, axes = pyplot.subplots(2, 3, sharex='all', sharey='all')
    fig.suptitle("Lyness-Kaganove diagrams")

    print("Lyness-Kaganove benchmark as described in Section 6.5 "
          "of P. Gonnet's thesis.\n"
          "Legend: success count (mean number of evaluations)\n")
    print("family \ rtol\t{}\t\t{}\t\t{}".format(*colnames))

    for (i, familyname), ((m, n), ax), *results in zip(
            enumerate(familynames), np.ndenumerate(axes), *datasets):
        if i:
            print()
        line = [f"f_{i}\t"]
        for j, (nevals, quantiles, successratios) in enumerate(results):
            for col in cols:
                line.append(f"{successratios[col]} ({nevals[col]:.0f})")
            print("\t".join(line))
            line = ["\t"]

            for k, (t, q) in enumerate(quantiles.items()):
                if j == m == n == 0:
                    label = f'{100*t:.0f}% quantile'
                elif k == m == 0 and n == 1:
                    label = dataset_names[j]
                else:
                    label = None
                ax.loglog(q, nevals, 'o-', markevery=10,
                          label=label)

        ax.set_title(f'$f_{i}(x) = {familyname}$')
        if m == 0 and n in (0, 1):
            ax.legend()
        if m == 1:
            ax.set_xlabel("Relative true error")
        if n == 0:
            ax.set_ylabel("Mean number of evaluations")


def main():
    VERSION = 'vquad benchmark dump version 0'

    #### Parse command line.
    examples = """Examples:
python3 -m vquad -
python3 -m vquad -a quadpack -o quadpack
python3 -m vquad quadpack vquad-original -
"""
    parser = argparse.ArgumentParser(
        'python3 -m vquad',
        description='Benchmark the integrator.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=examples)
    parser.add_argument('datasets', nargs='*')
    parser.add_argument('-a', '--algorithm', default='vquad',
                        choices=['vquad', 'quadpack'])
    parser.add_argument('-n', '--samples', type=int, default=200)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    # run_quadpack(args.samples, 10.0**np.linspace(0, -10, 101))

    #### Setup benchmark.
    if args.output or '-' in args.datasets:
        if args.algorithm == 'vquad':
            job = lk_job_vquad
        elif args.algorithm == 'quadpack':
            job = lk_job_quadpack
        else:
            raise RuntimeError('Unknown algorithm.')
        computed, tobesaved = itertools.tee(run(
            lk_runner, job,
            args.samples, 10.0**np.linspace(0, -10, 101), [0.5, 0.95]))

    datasets = []
    dataset_names = []
    #### Load any datasets.
    for fname in args.datasets:
        if fname == "-":
            datasets.append(computed)
            dataset_names.append('<{}>'.format(args.algorithm))
            continue
        with open(fname, 'rb') as f:
            version, loaded = pickle.load(f)
            if version != VERSION:
                raise RuntimeError("Invalid version of file " + fname)
            else:
                datasets.append(loaded)
                dataset_names.append(fname)

    #### Launch computation and display.
    if datasets:
        display(datasets, dataset_names, (f[0] for f in families),
                [30, 60, 90], ['1e-3', '1e-6', '1e-9'])

    #### Save benchmark.
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump((VERSION, list(tobesaved)), f)

    pyplot.show()
