"""
STEP ("Select the Easiest Point") is a scalar optimization algorithm
that minimizes a function by halving intervals over the bounded
space iteratively, each time selecting the interval with smallest
"difficulty".  The difficulty measure is curvature of x^2 function
crossing the interval boundary points and touching the supposed
(so-far-estimated) optimum; this curvature will be small for
intervals that have boundary points near the optimum.  The "smoother"
the function, the better this works.

http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=349896
http://www.applied-mathematics.net/optimization/Step.pdf

If you want to simply use STEP for straightforward scalar optimization,
you can invoke the ``step_minimize()`` function or invoke the algorithm
through ``scipy.optimize.minimize_scalar`` passing the return value
of ``step_minmethod()`` function as the method parameter.  Example:

>>> def f(x):
...     return (x - 2) * x * (x + 2)**2

>>> from step import step_minimize
>>> step_minimize(f, bounds=(-10, +10), maxiter=100)
{'fun': -9.91494958991847,
 'nit': 100,
 'success': True,
 'x': 1.2807846069335938}

>>> from step import step_minmethod
>>> import scipy.optimize as so
>>> so.minimize_scalar(f, bounds=(-10, +10), method=step_minmethod, options={'disp':False, 'maxiter':100})
     fun: -9.91494958991847
       x: 1.2807846069335938
 success: True
     nit: 100

You can also use the STEP class interface to single-step the algorithm,
possibly even tweaking its internal data structures between iterations.
We use that for multi-dimensional STEP.
"""

from __future__ import print_function
import copy
import numpy as np
from operator import itemgetter
import warnings
import pylab


class STEP(object):
    """
    This class implements the scalar STEP algorithm run in a piece-meal
    way that allows simple scalar optimization as well as tweaking
    of internal STEP data within multidimensional wrappers.

    Example:

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    >>> import step
    >>> optimize = step.STEP(f)
    >>> optimize.begin(bounds=(-10,10))
    >>> for i in range(100):
    ...     (x, y) = optimize.one_step()
    ...     if y is None: break
    ...     if optimize.fmin < 1e-8: break
    >>> print(optimize.xmin, optimize.fmin)

    """
    def __init__(self, fun, epsilon=1e-8, disp=False, tolx=1e-10, maxdiff=1e7, **options):
        """
        Set up a STEP algorithm instance on a particular function.
        This does not evaluate it in any way yet - to start optimization,
        call .begin(), then repeatedly .one_step().
        """
        self.fun = fun
        self.epsilon = epsilon
        self.disp = disp
        self.tolx = tolx
        self.maxdiff = maxdiff

        # These will be filled in begin()
        self.points = None
        self.values = None
        self.xmin = None
        self.fmin = None
        self.difficulty = None
        self.axis = None

        # Cached index of interval with lowest difficulty; this is set
        # and reused by .easiest_interval(), but must be cleared anytime
        # we touch the .difficulty[] array.
        self.easiest_i_cache = None

        # Warn about unrecognized options, but don't make them fatal.
        for o in options.keys():
            warnings.warn('STEP: Unrecognized option %s' % (o,))

    def begin(self, bounds, point0=None, axis=None):
        """
        Initialize the algorithm with particular global interval bounds
        and starting point (the middle of the interval by default).

        If the bounds are in multi-dimensional space, axis denotes the
        axis along which scalar optimization is performed (with the
        other dimensions held fixed).
        """
        self.axis = axis

        if point0 is None:
            point0 = (bounds[0] + bounds[1]) / 2.0
        else:
            point0 = np.array(point0)  # make a copy

        if axis is None:
            assert bounds[0] < point0 < bounds[1], point0
            self.points = np.array([bounds[0], point0, bounds[1]])
        else:
            assert bounds[0][axis] <= point0[axis] <= bounds[1][axis], point0[axis]
            self.points = np.array([np.array(point0), point0, np.array(point0)])
            self.points[0][axis] = bounds[0][axis]
            self.points[2][axis] = bounds[1][axis]
            # point0 might be at the boundary; in that case, re-halve it
            if self.points[0][axis] == self.points[1][axis] or \
               self.points[1][axis] == self.points[2][axis]:
                    self.points[1][axis] = (self.points[0][axis] + self.points[2][axis]) / 2.
        self.values = np.array([self.fun(p) for p in self.points])

        imin, self.fmin = min(enumerate(self.values), key=itemgetter(1))
        self.xmin = copy.copy(self.points[imin])

        self._recompute_difficulty()

        return (self.xmin, self.fmin)

    def one_step(self):
        """
        Perform one iteration of the STEP algorithm, which amounts to
        selecting the interval to halve, evaluating the function once
        there and updating the interval difficulties.

        Returns the (x, y) tuple for the selected point (this is NOT
        the currently found optimum; grab that from .xmin, .fmin).
        Returns (None, None) if no step could have been performed
        anymore (this signals the algorithm should be terminated).
        """

        i = self.easiest_interval()
        if i is None:
            # No suitable interval anymore
            return (None, None)

        # Split it into two
        newpoint = (self.points[i] + self.points[i+1]) / 2.0
        newvalue = self.fun(newpoint)
        self.points = np.insert(self.points, i+1, newpoint, axis=0)
        self.values = np.insert(self.values, i+1, newvalue, axis=0)
        self.difficulty[i] = np.nan
        self.difficulty = np.insert(self.difficulty, i+1, np.nan, axis=0)
        self.easiest_i_cache = None  # we touched .difficulty[]

        if newvalue < self.fmin:
            # New fmin, recompute difficulties of all intervals
            self.fmin = newvalue
            self.xmin = copy.copy(self.points[i+1])
            self._recompute_difficulty()
        else:
            # No fmin change, compute difficulties only of the two
            # new intervals
            self.difficulty[i:i+2] = self._interval_difficulty(self.points[i:i+3], self.values[i:i+3])
            # .easiest_i_cache already cleared

        return (newpoint, newvalue)

    def update_context(self, dx, dy):
        """
        This special-purpose function can be used for linear translation
        of all points and values between iterations.  We use this in ndstep
        when we run one STEP per dimension and assume linearly separable
        functions; when we find a new optimum along some dimension, we
        perform this translation in STEPs of other dimensions.
        """
        self.points += dx
        self.values += dy
        # Do not update xmin since it's reference to one of the points
        # above which we already updated.
        self.xmin += dx
        self.fmin += dy
        self._recompute_difficulty()

    def easiest_interval(self):
        """
        Find the easiest interval which is wide enough and return its
        index (i which corresponds to difficulty[i] and pair of
        points[i], points[i+1]).  This is mostly useful internally
        but may be also interesting for some dimension selection
        strategies in ndstep.
        """
        if self.easiest_i_cache is not None:
            # Reuse previously computed index
            return self.easiest_i_cache

        if self.axis is None:
            delta = self.points[1:] - self.points[:-1]
        else:
            delta = self.points[1:, self.axis] - self.points[:-1, self.axis]
        interval_wide_enough = delta >= self.tolx

        if self.disp:
            xx = np.array([x[0] for x in self.points])
            pylab.figure(1, figsize=(6,4))
            pylab.plot(xx, self.values, 'b')
            pylab.plot((xx[1:] + xx[:-1]) / 2, np.log(self.difficulty), 'k.')
            pylab.show()
            pylab.close()

        # This is the original version, which does not work for Python 3
        #idiff = filter(lambda (i, diff): interval_wide_enough[i], enumerate(self.difficulty))        
        idiff = []
        for i, diff in enumerate(self.difficulty):
            if interval_wide_enough[i]:
                idiff.append((i, diff))

        if len(idiff) == 0:
            return None  # We cannot split the interval more
        i, diff = min(idiff, key=itemgetter(1))
        if diff >= self.maxdiff:
            return None  # Even the best difficulty is too high

        if self.disp:
            print('Easiest interval %s: [%s, %s]' % (diff, self.points[i], self.points[i+1]))
        self.easiest_i_cache = i
        return i

    def _interval_difficulty(self, points, values):
        """
        Compute difficulty of intervals between the given list of points;
        for a mere pair of points, this is difficulty of just a single
        interval, of course.
        """

        # Recompute the second point coordinates with regards to the left (first)
        # point.
        if self.axis is None:
            x = points[1:] - points[:-1]
        else:
            x = points[1:, self.axis] - points[:-1, self.axis]

        # Some sanity checks:
        # We should differ in exactly one dimension
        if len(points) == 2:
            assert np.sum(points[1] != points[0]) == 1
        # Interval width should be positive and non-zero
        #assert np.all(x > 0)

        y = values[1:] - values[:-1]
        f = self.fmin - values[:-1] - self.epsilon

        # Curvature of parabole crossing [0,0], [x,y] and touching [?, f]
        a = (y - 2*f + 2*np.sqrt(f * (f - y))) / (x**2)
        return a

    def _recompute_difficulty(self):
        """
        Recompute the difficulty of all intervals.
        """
        self.difficulty = self._interval_difficulty(self.points, self.values)
        self.easiest_i_cache = None  # we touched .difficulty[]
        return self.difficulty


def step_minimize(fun, bounds, args=(), maxiter=100, callback=None, axis=None, point0=None, logf=None, staglimit=None, **options):
    """
    Minimize a given function within given bounds (a tuple of two points).

    The function can be multi-variate; in that case, you can pass numpy
    arrays as bounds, but you must also specify axis, as we still perform
    just scalar optimization along a specified axis.

    If staglimit is set to an integer, that constitutes the number of
    iterations to stop after if no better solution has been found.

    The logf file handle, if passed, is used for appending per-step
    evaluation information in text format.

    Example:

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    >>> from step import step_minimize
    >>> step_minimize(f, bounds=(-10, +10), maxiter=100)
    {'fun': -9.91494958991847,
     'nit': 100,
     'success': True,
     'x': 1.2807846069335938}

    """

    # Instantiate and fire off the STEP algorithm
    optimize = STEP(fun, **options)
    optimize.begin(bounds, point0=point0, axis=axis)

    niter = 0
    last_improvement = 0
    while niter < maxiter:
        if staglimit is not None and niter - last_improvement > staglimit:
            break

        y0 = optimize.fmin

        (x, y) = optimize.one_step()
        if y is None:
            break

        if logf:
            print("%d,%d,%e,%s,%d" % (axis, y < y0, y, ','.join(["%e" % xi for xi in x]), niter), file=logf)

        if y < y0:
            last_improvement = niter

        if callback is not None:
            if callback(optimize.xmin):
                break

        niter += 1

    return dict(fun=optimize.fmin, x=optimize.xmin, nit=niter,
                success=(niter > 1))


def step_minmethod(fun, **options):
    """
    A scipy.optimize.minimize_scalar method callable to use for minimization
    within the SciPy optimization framework.

    Example:

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    >>> from step import step_minmethod
    >>> import scipy.optimize as so
    >>> so.minimize_scalar(f, bounds=(-10, +10), method=step_minmethod(), options={'disp':False, 'maxiter':100})
         fun: -9.91494958991847
           x: 1.2807846069335938
     success: True
         nit: 100

    """
    from scipy import optimize

    del options['bracket']

    result = step_minimize(fun, **options)
    return optimize.OptimizeResult(**result)
