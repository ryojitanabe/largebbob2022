#!/usr/bin/env python
"""DEPRECATED: use rather `example_experiment2.py`

This is based on ``example_experiment.py`` (https://github.com/numbbo/coco/tree/master/code-experiments/build/python).

A Python script for the COCO experimentation module `cocoex`.

Usage from a system shell::

    python example_experiment.py bbob

runs a full but short experiment on the bbob suite. The optimization
algorithm used is determined by the `SOLVER` attribute in this file::

    python example_experiment.py bbob 20

runs the same experiment but with a budget of 20 * dimension
f-evaluations::

    python example_experiment.py bbob-biobj 1e3 1 20

runs the first of 20 batches with maximal budget of
1000 * dimension f-evaluations on the bbob-biobj suite.
All batches must be run to generate a complete data set.

Usage from a python shell:

>>> import example_experiment as ee
>>> ee.suite_name = "bbob-biobj"
>>> ee.SOLVER = ee.random_search  # which is default anyway
>>> ee.observer_options['algorithm_info'] = '"default of example_experiment.py"'
>>> ee.main(5, 1+9, 2, 300)  # doctest: +ELLIPSIS
Benchmarking solver...

runs the 2nd of 300 batches with budget 5 * dimension and at most 9 restarts.

Calling `example_experiment` without parameters prints this
help and the available suite names.

DEPRECATED: use rather `example_experiment2.py`
"""
from __future__ import absolute_import, division, print_function
try: range = xrange
except NameError: pass
import os, sys
import time
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from scipy import optimize # for tests with fmin_cobyla
from cocoex import Suite, Observer, log_level
del absolute_import, division, print_function

verbose = 1

try: import cma  # cma.fmin is a solver option, "pip install cma" installs cma
except: pass
try: from scipy.optimize import fmin_slsqp  # "pip install scipy" installs scipy
except: pass
try: range = xrange  # let range always be an iterator
except NameError: pass
try: time.process_time = time.clock
except: pass

from cocoex import default_observers  # see cocoex.__init__.py
from cocoex.utilities import ObserverOptions, ShortInfo, ascetime, print_flush
from cocoex.solvers import random_search


import getopt
import numpy as np
import sys

from step import STEP, step_minimize
from sqistep import SQISTEP, sqistep_minimize
from ndstep import ndstep_minimize
from ndstep_seq import ndstep_seq_minimize
from scipy_seq import scipy_seq_minimize

#import optuna

def _format_solution(res, optimum):
    """
    Return a string describing the solution described in res,
    relative to the optimum point.
    """
    delta = np.abs(res['x'] - optimum)
    closest_d = np.min(delta)
    farthest_d = np.max(delta)
    avg_d = np.average(delta)
    sd_d = np.std(delta)
    distance = np.linalg.norm(delta)
    solstr = 'y=%e  nit=% 6d  dx=(min=%e, max=%e, avg=%.3f (+- %.3f = %.3f), dist=%e)' % \
             (res['fun'], res['nit'],
              closest_d, farthest_d, avg_d, sd_d, avg_d + sd_d, distance)
    return solstr


class F4:
    """ Rastrigin-Bueche """
    def __init__(self, dim):
        self.dim = dim
        self.optimum = np.random.permutation(np.linspace(-4, 4, self.dim))

    def opt_y(self):
        return 0

    def __call__(self, xx):
        x = xx - self.optimum
        return 10 * (self.dim - np.sum(np.cos(2 * np.pi * x), -1)) + np.sum(x ** 2, -1)


class BBOB:
    """ A BBOB function """
    def __init__(self, dim, fid, iid):
        import bbobbenchmarks
        self.dim = dim
        (self.f, self.fopt) = bbobbenchmarks.instantiate(fid, iinstance=iid)

        self.f(np.zeros(dim))  # dummy eval so that we can grab xopt
        self.optimum = self.f.xopt

    def opt_y(self):
        return self.fopt

    def __call__(self, x):
        return self.f(x)


class BBOBFactory:
    """ A BBOB function factory """
    def __init__(self, fid, iid=1):
        self.fid = fid
        self.iid = iid

    def __call__(self, dim):
        return BBOB(dim, self.fid, self.iid)


class BBOBExperimentFactory:
    """ A BBOB function factory, in experiment setting (fev data recorded
    in COCO format for future evaluation and plotting using the BBOB
    toolchain. """
    def __init__(self, fid, iid, f):
        self.fid = fid
        self.iid = iid
        self.f = f

    def __call__(self, dim):
        bbob = BBOB(dim, self.fid, self.iid)
        self.f.setfun(bbob.f, bbob.fopt)
        bbob.f = self.f.evalfun  # XXX
        return bbob


def easiest_difficulty(o):
    i = o.easiest_interval()
    if i is not None:
        return o.difficulty[i]
    else:
        return o.maxdiff * 10


def easiest_sqi(o):
    if o is None: return np.Inf
    i = o.easiest_sqi_interval()
    if i is not None:
        return o.qfmin[i]
    else:
        return np.Inf


def easiest_difficulties(optimize):
    if normalize:
        return np.array([easiest_difficulty(o) / np.mean(o.difficulty) for o in optimize])
    else:
        return np.array([easiest_difficulty(o) for o in optimize])


def dimselect_random(fun, optimize, niter, min):
    return np.random.randint(len(optimize))

def dimselect_mindiff(fun, optimize, niter, min):
    return np.argmin(easiest_difficulties(optimize))

def dimselect_minsqi(fun, optimize, niter, min):
    # SQISTEP specific
    sqis = np.array([easiest_sqi(o) for o in optimize])
    bestsqi = np.argmin(sqis)
    if sqis[bestsqi] == np.inf:
        return dimselect_random(fun, optimize, niter, min)
    else:
        return bestsqi

def dimselect_maxdiff(fun, optimize, niter, min):
    return np.argmax(easiest_difficulties(optimize))

def dimselect_diffpd(fun, optimize, niter, min):
    # pd = easiest_difficulties(optimize)
    # pd = np.log(1 + easiest_difficulties(optimize))
    pd = np.log(easiest_difficulties(optimize))
    pd /= np.sum(pd)
    return np.random.choice(range(len(optimize)), p=pd)

def dimselect_rdiffpd(fun, optimize, niter, min):
    pd = np.log(easiest_difficulties(optimize))
    pd = 1. / pd
    pd /= np.sum(pd)
    return np.random.choice(range(len(optimize)), p=pd)


class DimSelectHistory:
    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def update(self, lastdim, min):
        (xmin, fmin) = min
        if lastdim >= 0:
            if fmin < self.lastfmin:
                self.hist[lastdim].append(self.lastfmin - fmin)
            else:
                self.hist[lastdim].append(0)
        if fmin < self.lastfmin:
            self.lastfmin = fmin

    def __call__(self, fun, optimize, niter, min):
        return np.argmax([np.mean(self.hist[i]) for i in range(len(self.hist))])

    def reset(self):
        self.hist = [[] for i in range(self.dim)]
        self.lastfmin = 1e10

class DimSelectHistoryRA:
    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def update(self, lastdim, min):
        # Record results of previous selection
        (xmin, fmin) = min
        if lastdim >= 0:
            if fmin < self.lastfmin:
                delta = self.lastfmin - fmin
            else:
                delta = 0
            if self.runmean[lastdim] is None:
                self.runmean[lastdim] = delta
            else:
                beta = 1/10  # 1/beta should be < stagiter
                self.runmean[lastdim] = beta * delta + (1 - beta) * self.runmean[lastdim]
        if fmin < self.lastfmin:
            self.lastfmin = fmin

    def __call__(self, fun, optimize, niter, min):
        # New selection
        return np.argmax([self.runmean[i] for i in range(len(self.runmean))])

    def reset(self):
        self.runmean = [None for i in range(self.dim)]
        self.lastfmin = 1e10

class DimSelectImprovementFreqRA:
    def __init__(self, dim):
        self.dim = dim
        self.reset()

    def reset(self):
        self.runmean = [1 for i in range(self.dim)]
        self.lastfmin = None

    def update(self, lastdim, min):
        # Record results of previous selection
        (xmin, fmin) = min
        if lastdim < 0:
            self.lastfmin = fmin
            return

        imp = 1 if fmin < self.lastfmin - 1e-8 else 0
        beta = 1e-1  # 1/beta should be < stagiter
        self.runmean[lastdim] = beta * imp + (1 - beta) * self.runmean[lastdim]
        if fmin < self.lastfmin:
            self.lastfmin = fmin

    def __call__(self, fun, optimize, niter, min):
        # New selection
        return np.argmax(self.runmean)


class DimSelectWrapper:
    """
    A generic wrapper around specific dimselect methods that
    performs some common tasks like updating history data,
    burn-in and epsilon-greedy exploration.
    """
    def __init__(self, options, dimselect):
        self.options = options
        self.dimselect = dimselect
        self.lastdim = -1

    def __call__(self, fun, optimize, niter, min):
        try:
            # For stateful dimselects
            self.dimselect.update(self.lastdim, min)
        except:
            pass

        if niter < len(optimize) * options['burnin']:
            # Round-robin - initially
            dim = niter % len(optimize)
        elif np.random.rand() <= self.options['egreedy']:
            # Random sampling - 1-epsilon frequently
            dim = np.random.randint(len(optimize))
        else:
            # The proper selection method
            dim = self.dimselect(fun, optimize, niter, min)

        self.lastdim = dim
        return dim

    def reset(self):
        self.lastdim = -1
        self.dimselect.reset()


def run_ndstep(logfname, minimize_function, options, f, stclass=STEP, minf=step_minimize):
    """
    A simple testcase for speed benchmarking, etc.

    We optimize the Rastrigin-Bueche function in 20D in range [-5,5]
    for maxiter iterations, using ndstep_minimize() with random restarts.
    """
    # Reproducible runs
    np.random.seed(options['seed'])

    dim = options['dim']
    #    f = options['f'](dim)
    logf = None  # open(logfname, mode='w')

    x0 = np.zeros(dim) - 5
    x1 = np.zeros(dim) + 5

    globres = dict(fun=np.Inf, x=None, nit=0, restarts=0, success=False)
    while globres['fun'] > 1e-8 and globres['nit'] < options['maxiter']:
        # Initial solution in a more interesting point than zero
        # to get rid of intrinsic regularities
        # When a minimization finishes, run a random restart then
        p0 = np.random.rand(dim) * 4 - 1

        res = minimize_function(lambda x: f(x),
                                bounds=(x0, x1), point0=p0,
                                maxiter=(options['maxiter'] - globres['nit']),
                                #callback=lambda x, y: y - f.opt_y() <= 1e-8,
                                logf=logf, dimselect=options['dimselect'],
                                stagiter=options['stagiter'],
                                force_STEP=options['force_STEP'],
                                force_Brent=options['force_Brent'],
                                split_at_pred=options['split_at_pred'],
                                posik_SQI=options['posik_SQI'],
                                stclass=stclass, minf=minf,
                                disp=options['disp'])
        #res['fun'] -= f.opt_y()        
        #print(_format_solution(res, f.optimum))
        # if res['fun'] < globres['fun']:
        #     globres['fun'] = res['fun']
        #     globres['x'] = res['x']
        #     globres['success'] = True
        globres['nit'] += res['nit']
        globres['restarts'] += 1
        try:
            # For stateful dimselects
            options['dimselect'].reset()
        except:
            pass

    # print('>>', globres)
    # print('>>', _format_solution(globres, f.optimum))
    return globres

def usage(err=2):
    print('Benchmark ndstep, ndstep_seq, ndsqistep, ndsqistep_seq, scipy_seq')
    print('Usage: test.py [-b BURNIN] [-f {f4,bFID}] [-d DIM] [-e {rr,random,mindiff,maxdiff,diffpd,rdiffpd}] [-g EPSILON] [-i MAXITER] [-s SEED] [-r REPEATS] [-t STAGITER] [-I FORCE_STEP_I] [-B FORCE_BRENT_I] [-p|-P] [-o] {nd[sqi]step,nd[sqi]step_seq,scipy_seq}')
    sys.exit(err)
    
def run(bbob_suite='bbob'):    
    method = "ndsqistep"

    ### input
    output_folder = method

    ### prepare
    suite = cocoex.Suite(bbob_suite, "", "")
    observer = cocoex.Observer(bbob_suite, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()
    
    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing
        run_step(problem, method)                           
        minimal_print(problem, final=problem.index == len(suite) - 1)        

def run_step(problem, method="ndsqistep"):
    options = {
        'f': F4,
        'dim': problem.dimension,
        'maxiter': 1e3,
        'seed': 43,
        'dimselect': None,
        'egreedy': 0,
        'burnin': 4,  # *D iters are spend systematically sampling first
        'stagiter': None,  # *D iters non-improving will cause a restart
        'force_STEP': 0,  # SQISTEP specific
        'force_Brent': 10,  # SQISTEP specific
        'split_at_pred': True,  # SQISTEP specific
        'posik_SQI': False,  # SQISTEP specific
        'disp': False,
    }

    options['f'] = problem
    options['maxiter'] *= options['dim']
   
    if options['dimselect'] == 'history':
        options['dimselect'] = DimSelectHistory(options['dim'])
    elif options['dimselect'] == 'historyRA':
        options['dimselect'] = DimSelectHistoryRA(options['dim'])
    elif options['dimselect'] == 'improvementFreqRA':
        options['dimselect'] = DimSelectImprovementFreqRA(options['dim'])

    if options['dimselect'] is not None:
        options['dimselect'] = DimSelectWrapper(options, options['dimselect'])

    if method == "ndstep":
        globres = run_ndstep('ndstep-log.txt', ndstep_minimize, options)
    elif method == "ndstep_seq":
        globres = run_ndstep('ndstep_seq-log.txt', ndstep_seq_minimize, options)
    elif method == "ndsqistep":
        #globres = run_ndstep('ndsqistep-log.txt', ndstep_minimize, options, stclass=SQISTEP)
        globres = run_ndstep('ndsqistep-log.txt', ndstep_minimize, options, problem, stclass=SQISTEP)
    elif method == "ndsqistep_seq":
        globres = run_ndstep('ndsqistep_seq-log.txt', ndstep_seq_minimize, options, minf=sqistep_minimize)
    elif method == "scipy_seq":
        globres = run_ndstep('scipy_seq-log.txt', scipy_seq_minimize, options)
    else:
        assert False


def default_observer_options(budget_=None, suite_name_=None, current_batch_=None):
    """return defaults computed from input parameters or current global vars
    """
    global budget, suite_name, number_of_batches, current_batch
    if budget_ is None:
        budget_ = budget
    if suite_name_ is None:
        suite_name_ = suite_name
    if current_batch_ is None and number_of_batches > 1:
        current_batch_ = current_batch
    opts = {}
    try:
        opts.update({'result_folder': '"%s_on_%s%s_budget%04dxD"'
                    % (SOLVER.__name__,
                       suite_name_,
                       "" if current_batch_ is None
                          else "_batch%03dof%d" % (current_batch_, number_of_batches),
                       budget_)})
    except: pass
    try:
        solver_module = '(%s)' % SOLVER.__module__
    except:
        solver_module = ''
    try:
        opts.update({'algorithm_name': SOLVER.__name__ + solver_module})
    except: pass
    return opts

# ===============================================
# loops over a benchmark problem suite
# ===============================================
def batch_loop(solver, suite, observer, budget,
               max_runs, current_batch, number_of_batches):
    """loop over all problems in `suite` calling
    `coco_optimize(solver, problem, budget * problem.dimension, max_runs)`
    for each eligible problem.

    A problem is eligible if ``problem_index + current_batch - 1``
    modulo ``number_of_batches`` equals ``0``.

    This distribution into batches is likely to lead to similar
    runtimes for the batches, which is usually desirable.
    """
    addressed_problems = []
    short_info = ShortInfo()
    for problem_index, problem in enumerate(suite):
        if (problem_index + current_batch - 1) % number_of_batches:
            continue
        observer.observe(problem)
        short_info.print(problem) if verbose else None
        runs = coco_optimize(solver, problem, budget * problem.dimension,
                             max_runs)
        if verbose:
            print_flush("!" if runs > 2 else ":" if runs > 1 else ".")
        short_info.add_evals(problem.evaluations + problem.evaluations_constraints, runs)
        problem.free()  # not necessary as `enumerate` tears the problem down
        addressed_problems += [problem.id]
    print(short_info.function_done() + short_info.dimension_done())
    short_info.print_timings()
    print("  %s done (%d of %d problems benchmarked%s)" %
           (suite_name, len(addressed_problems), len(suite),
             ((" in batch %d of %d" % (current_batch, number_of_batches))
               if number_of_batches > 1 else "")), end="")
    if number_of_batches > 1:
        print("\n    MAKE SURE TO RUN ALL BATCHES", end="")
    return addressed_problems

#===============================================
# interface: ADD AN OPTIMIZER BELOW
#===============================================
def coco_optimize(solver, fun, max_evals, max_runs=1e9):
    """`fun` is a callable, to be optimized by `solver`.

    The `solver` is called repeatedly with different initial solutions
    until either the `max_evals` are exhausted or `max_run` solver calls
    have been made or the `solver` has not called `fun` even once
    in the last run.

    Return number of (almost) independent runs.
    """
    range_ = fun.upper_bounds - fun.lower_bounds
    center = fun.lower_bounds + range_ / 2
    if fun.evaluations:
        print('WARNING: %d evaluations were done before the first solver call' %
              fun.evaluations)

    method = "ndsqistep"    
    run_step(fun, method)       

    return 1  # number of (almost) independent launches of `solver`

# ===============================================
# set up: CHANGE HERE SOLVER AND FURTHER SETTINGS AS DESIRED
# ===============================================
######################### CHANGE HERE ########################################
# CAVEAT: this might be modified from input args
suite_name = "bbob"  # always overwritten when called from system shell
                     # see available choices via cocoex.known_suite_names
budget = 2  # maxfevals = budget x dimension ### INCREASE budget WHEN THE DATA CHAIN IS STABLE ###
max_runs = 1e9  # number of (almost) independent trials per problem instance
number_of_batches = 1  # allows to run everything in several batches
current_batch = 1      # 1..number_of_batches
##############################################################################
# By default we call SOLVER(fun, x0), but the INTERFACE CAN BE ADAPTED TO EACH SOLVER ABOVE
SOLVER = random_search
SOLVER.__name__ = "ndsqistep"
# SOLVER = optimize.fmin_cobyla
# SOLVER = my_solver # SOLVER = fmin_slsqp # SOLVER = cma.fmin
suite_instance = "" # "year:2016"
suite_options = ""  # "dimensions: 2,3,5,10,20 "  # if 40 is not desired
# for more suite options, see http://numbbo.github.io/coco-doc/C/#suite-parameters
observer_options = ObserverOptions({  # is (inherited from) a dictionary
                    'algorithm_info': '"ndsqistep"', # CHANGE/INCOMMENT THIS!
                    # 'algorithm_name': '',  # default already provided from SOLVER name
                    # 'result_folder': '',  # default already provided from several global vars
                   })
######################### END CHANGE HERE ####################################

# ===============================================
# run (main)
# ===============================================
def main(budget=budget,
         max_runs=max_runs,
         current_batch=current_batch,
         number_of_batches=number_of_batches):
    """Initialize suite and observer, then benchmark solver by calling
    ``batch_loop(SOLVER, suite, observer, budget,...``
    """
    suite = Suite(suite_name, suite_instance, suite_options)

    observer_name = default_observers()[suite_name]
    # observer_name = another observer if so desired
    observer_options.update_gracefully(default_observer_options())
    observer = Observer(observer_name, observer_options.as_string)

    print("Benchmarking solver '%s' with budget=%d*dimension on %s suite, %s"
          % (' '.join(str(SOLVER).split()[:2]), budget,
             suite.name, time.asctime()))
    if number_of_batches > 1:
        print('Batch usecase, make sure you run *all* %d batches.\n' %
              number_of_batches)
    t0 = time.process_time()
    batch_loop(SOLVER, suite, observer, budget, max_runs,
               current_batch, number_of_batches)
    print(", %s (%s total elapsed time)." %
            (time.asctime(), ascetime(time.process_time() - t0)))
    print('Data written to folder', observer.result_folder)
    print('To post-process the data call \n'
          '    python -m cocopp %s \n'
          'from a system shell or \n'
          '    cocopp.main("%s") \n'
          'from a python shell' % (2 * (observer.result_folder,)))

# ===============================================
if __name__ == '__main__':
    """read input parameters and call `main()`"""
    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            print("Recognized suite names: " + str(cocoex.known_suite_names))
            sys.exit(0)
    suite_name = sys.argv[1]
    if suite_name not in cocoex.known_suite_names:
        print('WARNING: "%s" not in known names %s' %
                (suite_name, str(cocoex.known_suite_names)))
    if len(sys.argv) > 2:
        budget = float(sys.argv[2])
    if len(sys.argv) > 3:
        current_batch = int(sys.argv[3])
    if len(sys.argv) > 4:
        number_of_batches = int(sys.argv[4])
    if len(sys.argv) > 5:
        messages = ['Argument "%s" disregarded (only 4 arguments are recognized).' % sys.argv[i]
            for i in range(5, len(sys.argv))]
        messages.append('See "python example_experiment.py -h" for help.')
        raise ValueError('\n'.join(messages))
    main(budget, max_runs, current_batch, number_of_batches)
