# control-sim

## Background

This project provides code in C, supplemented with a MATLAB interface, for obtaining an approximate solution of generalized continuous-time algebraic Riccati equations ([Damm, 2004][1]) of the form

*A' X* + *X A* + *G' X G* - *X B X* + *Q* = 0, 

where *X* is an unknown positive-definite *n* x *n* matrix, while *A*, *B*, *G* and *Q* are given *n* x *n* matrices, such that *A* is stable, *B* is idempotent and positive-semidefinite, of rank *m < n*, *Q* is positive-definite, and *G* has, in a certain sense, a sufficiently small norm.

The numerical scheme for the solution of the above nonlinear matrix equation is based on a version of Newton's method, with exact line search ([Sima-Benner, 2006][2]). Given an approximation *Xk* of the true solution *X*, the next approximation will be of the form 

*Xk* + *t dX*, 

where *t* is a suitably chosen real number, and *dX* is the solution of the associated linearized equation, i.e. the generalized Lyapunov equation

(*A* - *B Xk*)' *dX* + *dX* (*A* - *B Xk*) + *G' dX G* + *Yk* = 0,

where 

*Yk* = *A' Xk* + *Xk A* + *G' Xk G* - *Xk B Xk* + *Q* 

is the current residual. The latter matrix equation is solved directly, as a system of *n*(*n*+1)/2 linear equations in the entries of the unknown symmetric matrix *dX*.

## Main Source Files
* [`src/gencare.c`][3] &nbsp; Solution to the generalized Riccati equation
* [`src/genlyap.c`][4] &nbsp; Solution to the generalized Lyapunov equation
* [`src/random.c`][5] &nbsp;&nbsp;&nbsp; Generation of random matrices with specified properties


## Remarks/Caveats

* No mathematical error analysis of the numerical scheme has been carried out by the author.
* Due to the naive method for solution of the generalized Lyapunov equations arising at each iteration of the algorithm outlined above, the provided code is not suitable for problems involving large matrices.
* The current version of the code uses C99-style variable length arrays as local variables for storing matrices of order *O*(*n*^2), and hence is limited by the size of the call-stack on the machine where they are to be executed.
* The code relies on the Intel Math Kernel Library (MKL) for calls to BLAS/LAPACK routines, as well as on the Subroutine Library in Systems and Control Theory (SLICOT), version 5.6. The source code for SLICOT is available upon request from http://slicot.org, but is not included in this repository, due to licensing restrictions.
* On the author's test machine, the C99 code in this repository and the Fortran 77 code from SLICOT were compiled with the Intel C++/Fortran 2018 compilers for Windows. The MATLAB wrapper functions for the C code were compiled with MATLAB R2018a for Windows. Only limited testing of the code has been carried out.

[1]:https://link.springer.com/book/10.1007/b10906
[2]:https://ieeexplore.ieee.org/document/4124845
[3]:/src/gencare.c
[4]:/src/genlyap.c
[5]:/src/random.c
