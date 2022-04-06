"""
Hybrid STEP ("Select the Easiest Point") and Successive Quadratic
Interpolation algorithm.  It divides the function domain to intervals
which are used for three-point quadratic interpolation or STEP
depending on potential improvements.

If you want to simply use SQISTEP for straightforward scalar optimization,
you can invoke the ``sqistep_minimize()`` function or invoke the algorithm
through ``scipy.optimize.minimize_scalar`` passing the return value
of ``sqistep_minmethod()`` function as the method parameter.  Example:

>>> def f(x):
...     return (x - 2) * x * (x + 2)**2

>>> from sqistep import sqistep_minimize
>>> sqistep_minimize(f, bounds=(-10, +10), maxiter=100)
{'fun': -9.9149495906555423,
 'nit': 100,
 'success': True,
 'x': 1.2807728342781046}

>>> from sqistep import sqistep_minmethod
>>> import scipy.optimize as so
>>> so.minimize_scalar(f, bounds=(-10, +10), method=sqistep_minmethod, options={'disp':False, 'maxiter':100})
     fun: -9.9149495906555423
       x: 1.2807728342781046
     nit: 100
 success: True

You can also use the SQISTEP class interface to single-step the algorithm,
possibly even tweaking its internal data structures between iterations.
We use that for multi-dimensional SQISTEP.
"""

from __future__ import print_function
import copy
import numpy as np
from operator import itemgetter

from step import STEP


class SQISTEP(STEP):
    """
    This class implements the scalar SQISTEP algorithm run in a piece-meal
    way that allows simple scalar optimization as well as tweaking
    of internal SQISTEP data within multidimensional wrappers.

    Example:

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    >>> import sqistep
    >>> optimize = sqistep.SQISTEP(f)
    >>> optimize.begin(bounds=(-10,10))
    >>> for i in range(100):
    ...     (x, y) = optimize.one_step()
    ...     if y is None: break
    ...     if optimize.fmin < 1e-8: break
    >>> print(optimize.xmin, optimize.fmin)

    """
    def __init__(self, fun, epsilon=1e-8, disp=False, tolx=1e-10, maxdiff=1e7, force_STEP=0, force_Brent=10, split_at_pred=True, posik_SQI=False, **options):
        """
        Set up a SQISTEP algorithm instance on a particular function.
        This does not evaluate it in any way yet - to start optimization,
        call .begin(), then repeatedly .one_step().

        If force_STEP = N > 0, every N iterations STEP is forcibly invoked
        instead of (potentially) SQI.  SQI may exhibit slow convergence
        when the function is quite non-quadratic.  On the other hand, if
        force_Brent = N > 0, every N iterations Brent (or SQI) is forcibly
        invoked instead of STEP when a suitable NIP is available, even if
        no improvement is predicted.

        split_at_pred set to False will keep SQI in use for interval selection,
        but when an interval is selected, it is sampled in its half, not at the
        predicted minimum point.  (This might help some convergence issues when
        the minimum is continuously predicted very close to one of endpoints.)

        posik_SQI set to True will cause the Posik method of SQI to be used
        instead of the Brent method.
        """
        super(SQISTEP, self).__init__(fun, epsilon, disp, tolx, maxdiff, **options)

        self.force_STEP = force_STEP
        self.force_Brent = force_Brent
        self.split_at_pred = split_at_pred
        self.posik_SQI = posik_SQI

        # Will be filled in .begin():
        self.qxmin = None  # xmin over neighboring interval pairs (NIP), quadratic interpolation
        self.qfmin = None  # fmin over neighboring interval pairs (NIP), quadratic interpolation

    def begin(self, bounds, point0=None, axis=None):
        """
        Initialize the algorithm with particular global interval bounds
        and starting point (the middle of the interval by default).

        If the bounds are in multi-dimensional space, axis denotes the
        axis along which scalar optimization is performed (with the
        other dimensions held fixed).
        """
        super(SQISTEP, self).begin(bounds, point0, axis)

        # We make the qxmin, qfmin arrays as big as the points arrays
        # even though in fact we use only N-2 elements instead of N.
        self.qxmin = np.array([self.mdpoint(None), self.mdpoint(None), self.mdpoint(None)])
        self.qfmin = np.array([np.Inf, np.Inf, np.Inf])
        self._update_qmins(0)
        self.itercnt = 0

        return (self.xmin, self.fmin)

    def easiest_sqi_interval(self):
        """
        Easiest sequential-quadratic-interpolation NIP, if there
        is any and the estimate is better than xmin - epsilon.

        Note that to use NIP index as interval index, you should check
        if the qxmin[i] is smaller or larger than points[i+1] - the
        NIP covers three points, not just two!
        """
        # Do not take guesses that are too near one of the interval
        # pair boundaries
        if self.axis is None:
            qxmin_suitable = np.logical_and(self.qxmin - self.points > self.tolx,
                                            np.roll(self.points, -2) - self.qxmin > self.tolx)
        else:
            qxmin_suitable = np.logical_and(self.qxmin[:,self.axis] - self.points[:,self.axis] > self.tolx,
                                            np.roll(self.points[:,self.axis], -2) - self.qxmin[:,self.axis] > self.tolx)
        # This is the original version, which does not work for Python 3            
        #iqfmin = filter(lambda (i, qfmin): qxmin_suitable[i], enumerate(self.qfmin))

        iqfmin = []
        for i, qfmin in enumerate(self.qfmin):
             if qxmin_suitable[i]:
                 iqfmin.append((i, qfmin))
        
        if len(iqfmin) == 0:
            # print('stop split')
            return None  # We cannot split further
        i, qfmin = min(iqfmin, key=itemgetter(1))
        if qfmin > self.fmin - self.epsilon and (self.force_Brent == 0 or self.itercnt % self.force_Brent > 0):
            # print('%s > %s' % (qfmin, self.fmin - self.epsilon))
            return None  # Even the best estimate is too high
        return i

    def mdpoint(self, x):
        """
        Generate a multi-dimensional point from scalar x.
        """
        if self.axis is None:
            return x
        mdx = np.array(self.points[0])
        mdx[self.axis] = x
        return mdx

    def one_step(self):
        """
        Perform one iteration of the SQISTEP algorithm, which amounts to
        selecting the interval to halve, evaluating the function once
        there and updating the interval difficulties.

        Returns the (x, y) tuple for the selected point (this is NOT
        the currently found optimum; grab that from .xmin, .fmin).
        Returns (None, None) if no step could have been performed
        anymore (this signals the algorithm should be terminated).
        """

        self.itercnt += 1

        # Try SQI - except on some iterations, maybe forcibly do STEP
        npi_i = None
        if self.force_STEP == 0 or self.itercnt % self.force_STEP > 0:
            npi_i = self.easiest_sqi_interval()
        if npi_i is not None:
            newpoint = self.qxmin[npi_i]
            newvalue = self.qfmin[npi_i]

            # Convert NIP index to interval index
            if ((self.axis is None and newpoint > self.points[npi_i+1]) or
               (self.axis is not None and newpoint[self.axis] > self.points[npi_i+1][self.axis])):
                i = npi_i + 1
                if self.disp:
                    print('SQI chose interval %s: x=[%s %s {%s} %s] y=[%s %s {%s} %s]' %
                          (i, self.points[npi_i], self.points[npi_i+1], newpoint, self.points[npi_i+2],
                           self.values[npi_i], self.values[npi_i+1], newvalue, self.values[npi_i+2]))
            else:
                i = npi_i
                if self.disp:
                    print('SQI chose interval %s: x=[%s {%s} %s %s] y=[%s {%s} %s %s]' %
                          (i, self.points[npi_i], newpoint, self.points[npi_i+1], self.points[npi_i+2],
                           self.values[npi_i], newvalue, self.values[npi_i+1], self.values[npi_i+2]))

            if not self.split_at_pred:
                # Actually split in half instead, best point
                # predictions may be overbiased for quadratic
                # function
                newpoint = (self.points[i] + self.points[i+1]) / 2.0

        # Try STEP
        else:
            i = self.easiest_interval()
            if i is None:
                # No suitable interval anymore
                return (None, None)

            # Split it in half
            newpoint = (self.points[i] + self.points[i+1]) / 2.0
            if self.disp:
                print('STEP chose interval %s: [%s, %s], point %s' % (i, self.points[i], self.points[i+1], newpoint))

        newvalue = self.fun(newpoint)
        self.points = np.insert(self.points, i+1, newpoint, axis=0)
        self.values = np.insert(self.values, i+1, newvalue, axis=0)
        self.difficulty[i] = np.nan
        self.difficulty = np.insert(self.difficulty, i+1, np.nan, axis=0)
        self.easiest_i_cache = None  # we touched .difficulty[]

        self.qxmin = np.insert(self.qxmin, i+1, self.mdpoint(np.nan), axis=0)
        self.qfmin = np.insert(self.qfmin, i+1, np.Inf, axis=0)
        if i > 0:
            self._update_qmins(i-1)
        if i < np.size(self.points, axis=0) - 2:
            self._update_qmins(i)
        if i+1 < np.size(self.points, axis=0) - 2:
            self._update_qmins(i+1)

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
        of all points and values between iterations.  We use this in ndsqistep
        when we run one SQISTEP per dimension and assume linearly separable
        functions; when we find a new optimum along some dimension, we
        perform this translation in SQISTEPs of other dimensions.
        """
        super(SQISTEP, self).update_context(dx, dy)

        self.qxmin += dx
        self.qfmin += dy

    def _update_qmins(self, i):
        """
        Update quadratic interpolations of NIP starting at point index i.
        """
        self.qxmin[i], self.qfmin[i] = self._nip_qinterp(self.points[i:i+3], self.values[i:i+3])

    def _nip_qinterp(self, points, values):
        """
        Make a quadratic interpolation of the minimum in a neighboring
        pair of intervals (i.e. three points on the curve).  Returns
        the xmin and fmin of this NIP.
        """
        if values[1] >= values[0] or values[1] >= values[2]:
            # This interpolation will not result in a smaller minimum
            # than the sampled points
            return (None, np.Inf)

        if self.axis is None:
            x0 = points[0]
            xr = points[1]
            xs = points[2]
        else:
            x0 = points[0][self.axis]
            xr = points[1][self.axis]
            xs = points[2][self.axis]
        y0 = values[0]
        yr = values[1]
        ys = values[2]

        # Compute a,b for ym estimate
        # XXX: Use the coefficients below instead
        XR = xr - x0
        XS = xs - x0
        YR = yr - y0
        YS = ys - y0
        a = (XR * YS - XS * YR) / (XR * XS * (XS - XR))
        b = (YR / XR) - (XR * YS - XS * YR) / (XS * (XS - XR))
        if self.posik_SQI:
            # This is the original computation by Petr Posik
            xm = x0 - b / (2*a)
            ym = y0 - b**2 / (4*a)
            return (self.mdpoint(xm), ym)

        xa = x0 if y0 > ys else xs  # worse of boundaries
        ya = y0 if y0 > ys else ys  # worse of boundaries
        xb = x0 if y0 <= ys else xs  # better of boundaries
        yb = y0 if y0 <= ys else ys  # better of boundaries

        R = (xr-xb) * (yr-ya)
        Q0 = (xr-xa) * (yr-yb)
        P = (xr-xa) * Q0 - (xr-xb) * R
        Q = 2 * (Q0 - R)
        if Q > 0:
            P = -P
        Q = abs(Q)

        # We use the Brent algorithm condition (only with non-historical
        # etemp; see Numerical Recipes Ch10.2) to decide whether to accept
        # the new point:
        step = P/Q
        etemp = min(xr - x0, xs - xr)  # delta between middle and boundary point
        if abs(step) < 0.5 * etemp and xr + step > x0 and xr + step < xs:
            # print('accepting %s,%s,%s step %s sample %s' % (x0, xr, xs, step, xr + step))
            d = step
        else:
            # Take a golden ratio step instead
            # print('rejecting %s,%s,%s step %s sample %s' % (x0, xr, xs, step, xr + step))
            # return (None, np.Inf)
            if xr >= (xs+x0) / 2:
                d = 0.3819660 * (x0-xr)
            else:
                d = 0.3819660 * (xs-xr)

        xm = xr + d
        ym = y0 - b**2 / (4*a)

        return (self.mdpoint(xm), ym)


def sqistep_minimize(fun, bounds, args=(), maxiter=100, callback=None, axis=None, point0=None, logf=None, staglimit=None, **options):
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

    >>> from sqistep import sqistep_minimize
    >>> sqistep_minimize(f, bounds=(-10, +10), maxiter=100)
    {'fun': -9.9149495906555423,
     'nit': 100,
     'success': True,
     'x': 1.2807728342781046}
    """

    # Instantiate and fire off the SQISTEP algorithm
    optimize = SQISTEP(fun, **options)
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


def sqistep_minmethod(fun, **options):
    """
    A scipy.optimize.minimize_scalar method callable to use for minimization
    within the SciPy optimization framework.

    Example:

    >>> def f(x):
    ...     return (x - 2) * x * (x + 2)**2

    >>> from sqistep import sqistep_minmethod
    >>> import scipy.optimize as so
    >>> so.minimize_scalar(f, bounds=(-10, +10), method=sqistep_minmethod(), options={'disp':False, 'maxiter':100})
         fun: -9.9149495906555423
           x: 1.2807728342781046
         nit: 100
     success: True

    """
    from scipy import optimize

    del options['bracket']

    result = sqistep_minimize(fun, **options)
    return optimize.OptimizeResult(**result)
