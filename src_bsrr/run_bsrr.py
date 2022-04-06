"""
This is based on ``test.py`` (https://github.com/pasky/step/blob/master/test.py) and ``example_experiment_for_beginners.py`` (https://github.com/numbbo/coco/tree/master/code-experiments/build/python).
"""
from __future__ import print_function

import getopt
import numpy as np
import sys

from step import STEP, step_minimize
from sqistep import SQISTEP, sqistep_minimize
from ndstep import ndstep_minimize
from ndstep_seq import ndstep_seq_minimize
from scipy_seq import scipy_seq_minimize

#import numpy as np
import cocoex  # only experimentation module
import sys
import logging
import os
import random
# import click
# import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        'maxiter': 1e4,
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
            
if __name__ == '__main__':
    run_id = 0
    np.random.seed(seed=run_id)
    random.seed(run_id)
    
    run(bbob_suite='bbob-largescale')
    # run(bbob_suite='bbob')    
