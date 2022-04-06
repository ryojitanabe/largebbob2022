# Benchmarking the Hooke-Jeeves Method, MTS-LS1, and BSrr on the Large-scale BBOB Function Set

This repository provides the code to reproduce results shown in the following paper.

> Ryoji Tanabe, **Benchmarking the Hooke-Jeeves Method, MTS-LS1, and BSrr on the Large-scale BBOB Function Set**, submitted.

The code highly depends on the COCO software:

> Nikolaus Hansen, Anne Auger, Raymond Ros, Olaf Mersmann, Tea Tusar, and Dimo Brockhoff, **COCO: A Platform for Comparing Continuous Optimizers in a Black-Box Setting**, Optimization Methods and Software, 36(1): 114-144 (2021), [link](https://arxiv.org/abs/1603.08785)

# Requirements

This code require Python (=>3.8) and a C compiler. I compiled this code with GNU gcc version 11.1.0 on the Ubuntu 18.04.

# How to run the Hooke-Jeeves method and MTS-LS1

After moving to ``src``, please compile the C programs as follows:

```
$ cd src_c
$ make
```

The C files in ``src`` were derived from [the C files provided by the COCO platform](https://github.com/numbbo/coco/tree/master/code-experiments/build/c). Since experiments on the high-dimensional functions are very time-consuming (especially for $f_{21}$), I slightly modified the original ``example_experiment.c`` so that I can run the program on each BBOB function individually as in the batch mode in the Python version (``example_experiment.py``).
The following command runs the Hooke-Jeeves method on the $f_{2}$ with 20, 40, 80, 160, 320, and 640 dimensions in the large-scale BBOB function set:

```
./example_experiment -fun_id 2
```

To run MTS-LS1, please comment out the corresponding lines for the Hooke-Jeeves method (lines 228-236) in the ``example_experiment`` function. Then, please uncomment the corresponding lines for MTS-LS1 (lines 238-247).

# How to run BSrr

After moving to ``src_bsrr``, please run ``run_bsrr.py`` as follows:

```
$ cd src_bsrr
$ python run_bsrr.py
```

All files in ``src_bsrr`` are based on the code provided by the first author of BSrr (https://github.com/pasky/step).

> Petr Baudis, Petr Pos{\'{\i}}k: Global Line Search Algorithm Hybridized with Quadratic Interpolation and Its Extension to Separable Functions. GECCO 2015: 257-264

We slightly modified the original code so that we can run it with Python 3 and the new version of the COCO platform.
As reported in our BBOB workshop paper, the Python code is time-consuming for large dimensions. We suggest using  the batch mode as follows, where it is based on ``example_experiment.py`` (https://github.com/numbbo/coco/tree/master/code-experiments/build/python). The following command runs the 77-th of 100 batches:

```
$ python batched\_step\_run.py bbob-largescale 1e3 77 100
```
