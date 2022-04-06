"""
STEP is a scalar optimization algorithm.  This module provides
an ``ndstep_minimize`` function that tries to apply it to multivariate
optimization nevertheless.

The approach is to find an optimum in each dimension by a separate STEP
algorithm but the steps along different dimensions are interleaved and
improving solutions along one dimensions are propagated to STEP
intervals in all other dimensions.

Example:

>>> def f(x):
...     return np.linalg.norm(x) ** 2
>>> x0 = np.array([-3, -3, -3])
>>> x1 = np.array([+1, +2, +3])

>>> from ndstep import ndstep_minimize
>>> ndstep_minimize(f, bounds=(x0, x1), maxiter=1000)
{'fun': 3.637978807091713e-12,
 'nit': 1000,
 'success': True,
 'x': array([  0.00000000e+00,   1.90734863e-06,   0.00000000e+00])}

"""

from __future__ import print_function

import numpy as np

from step import STEP


def ndstep_minimize(fun, bounds, args=(), maxiter=2000, callback=None,
                    point0=None, dimselect=None, stagiter=None, logf=None,
                    stclass=STEP, **options):
    """
    Minimize a given multivariate function within given bounds
    (a tuple of two points).

    Each dimension is optimized by a separate STEP algorithm but the
    steps along different dimensions are interleaved and improving
    solutions along one dimensions are propagated to STEP intervals
    in all other dimensions.

    The stopping condition is either maxiter total iterations or when
    stagiter optimization steps are done without reaching an improvement
    (whichever comes first).  By default, stagiter is 2000.

    Dimensions are selected using a round-robin strategy by default.
    You can pass a custom dimension selection function that is called
    as dimselect(fun, [step...], niter, min=(xmin, fmin)):

    >>> # Rastrigin-Bueche
    >>> def f(x): return 10 * (20 - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)
    >>> x0 = np.ones(20) - 5
    >>> x1 = np.ones(20) + 5
    # Random dimension choice
    >>> ndstep_minimize(f, bounds=(x0, x1), maxiter=2000,
    ...     dimselect=lambda fun, optimize, niter, min:
    ...         np.random.permutation(range(len(optimize)))[0])
    # Easiest dimensions choice
    >>> p0 = np.random.rand(20)
    >>> ndstep_minimize(f, bounds=(x0, x1), point0=p0, maxiter=2000,
    ...     dimselect=lambda fun, optimize, niter, min:
    ...         np.argmin([o.difficulty[o.easiest_interval()] for o in optimize])
    ...             if niter >= len(optimize) * 4
    ...             else niter % len(optimize))

    The callback, if passed, is called with the current optimum hypothesis
    (x, y) every 10*DIM iterations; if it returns True, the optimization
    run is stopped.

    The logf file handle, if passed, is used for appending per-step
    evaluation information in text format.

    See the module description for an example.
    """

    dim = np.shape(bounds[0])[0]
    disp = options.get('disp', False)
    if stagiter is None:
        stagiter = 2000
    callback_interval = 10 * dim

    xmin = np.array(point0)
    fmin = fun(point0)

    optimize = [stclass(fun, **options) for i in range(dim)]
    for i in range(dim):
        (x, y) = optimize[i].begin(bounds, point0=xmin, axis=i)

        if y < fmin:
            for j in range(i):
                optimize[j].update_context(x - xmin, y - fmin)
            xmin = np.array(x)
            fmin = y

    niter = -1
    niter_callback = callback_interval
    last_improvement = 0  # #iter that last brought some improvement
    while True:
        niter += 1

        # Test stopping conditions
        if maxiter is not None and niter >= maxiter:
            # Too many iterations
            break
        if last_improvement < niter - stagiter:
            # No improvement for the last #dim iterations
            break

        # Pick the next dimension to take a step in
        if dimselect is None:
            # By default, use round robin
            i = niter % dim
        else:
            i = dimselect(fun, optimize, niter, min=(xmin, fmin))

        if optimize[i] is None:
            continue

        if disp: print('-----------------------', i)
        x0 = np.array(optimize[i].xmin)
        y0 = optimize[i].fmin
        (x, y) = optimize[i].one_step()
        if disp: print(x, y)
        if y is None:
            optimize[i] = None
            continue

        if logf:
            print("%d,%d,%e,%s,%d" % (i, y < y0, y, ','.join(["%e" % xi for xi in x]), niter), file=logf)

        if y < y0:
            # We found an improving solution, shift the "context" on
            # all other axes
            if disp: print('improving solution!')
            xmin = x
            fmin = y
            for j in range(dim):
                if i == j or optimize[j] is None:
                    continue
                optimize[j].update_context(x - x0, y - y0)
            last_improvement = niter

        if callback is not None and niter >= niter_callback:
            if callback(xmin, fmin):
                break
            niter_callback = niter + callback_interval

    return dict(fun=fmin, x=xmin, nit=niter,
                success=(niter > 1))


def ndstep_minmethod(fun, x0, **options):
    """
    A scipy.optimize.minimize method callable to use for minimization
    within the SciPy optimization framework.

    Example:

    >>> def f(x):
    ...     return np.linalg.norm((x - 2) * x * (x + 2)**2)

    >>> from ndstep import ndstep_minmethod
    >>> import scipy.optimize as so
    >>> p0 = np.random.rand(3)
    >>> so.minimize(f, p0, bounds=((-3,+1), (-3,+2), (-3,+3)), method=ndstep_minmethod, options={'disp':False, 'maxiter':2000})
         fun: 9.3932069273767208e-16
           x: array([-0.99996631,  2.        ,  2.0007044 ])
         nit: 97
     success: True

    """
    from scipy import optimize

    for k in ('hess', 'hessp', 'jac', 'constraints'):
        del options[k]

    # Rearrange the bounds to a more sensible form
    bounds = [[], []]
    for (a, b) in options['bounds']:
        bounds[0].append(a)
        bounds[1].append(b)
    options['bounds'] = [np.array(bounds[0]), np.array(bounds[1])]

    # Basinhopping can pick a starting point outside of our bounds;
    # what about we move them?
    if np.any(x0 < options['bounds'][0]):
        options['bounds'][0] = np.array(x0)
    elif np.any(x0 > options['bounds'][1]):
        options['bounds'][1] = np.array(x0)
    bounds = [np.min([options['bounds'][0], options['bounds'][1]], axis=0),
              np.max([options['bounds'][0], options['bounds'][1]], axis=0)]
    options['bounds'] = bounds

    # Drop the second parameter of callback
    orig_callback = options['callback']
    options['callback'] = lambda x, f: orig_callback(x)

    result = ndstep_minimize(fun, point0=x0, **options)
    return optimize.OptimizeResult(**result)
