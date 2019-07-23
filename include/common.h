#ifndef COMMONDECL_H
#define COMMONDECL_H

#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef double _Complex DCOMPLEX;
#define MKL_Complex16 DCOMPLEX
#include "mkl.h"
#include "omp.h" 

#define ROW(A, i)        ((A) + (i))
#define COL(A, ld, j)    ((A) + (ld) * (j))
#define BLK(A, ld, i, j) ((A) + (i) + (ld) * (j))
#define ELT(A, ld, i, j) (A)[(i) + (ld) * (j)]

/* GENLYAP */
void lyap_solve(
	char trans, 
	MKL_INT n, 
	const double *restrict A, 
	const double *restrict Y, 
	double *X);

void gen_lyap_mat(
	MKL_INT n, 
	const double *restrict A, 
	const double *restrict G, 
	double *restrict t);

void gen_lyap_solve_direct( 
	MKL_INT n, 
	double alpha,
	const double *restrict A, 
	const double *restrict G, 
	const double *restrict Y,
	double *restrict T,
	double *restrict X);

/* GENCARE */
void gen_care_eval(
	char trans,
	MKL_INT n,
	double gamma, 
	const double *restrict A,
	const double *restrict G,
	const double *restrict B,
	const double *restrict Q,
	const double *restrict X,
	double *restrict Y);

void gen_care_solve_newton(
	MKL_INT kmax,
	MKL_INT n,
	double gamma, 
	const double *restrict A,
	const double *restrict G,
	const double *restrict B,
	const double *restrict Q,
	double *restrict T,
	double *restrict X,
	MKL_INT *nstep);

/* MINCOST */
void min_cost_ngrad(
	MKL_INT kmax,
	MKL_INT n,
	double gamma,
	const double *restrict A,
	const double *restrict G,
	const double *restrict B,
	const double *restrict Q,
	const double *restrict L,
	double *restrict T,
	double *restrict ngradJ);

/* RANDOM */
void gen_seed(MKL_INT iseed[static 4]);

void rand_eigen(
	MKL_INT n, 
	MKL_INT nr, 
	double remin, 
	double remax, 
	double *x, 
	char *type,
	MKL_INT *iseed);

void rand_mat(
	MKL_INT n,  
	double remin, 
	double remax, 
	double *A,
	MKL_INT *iseed);

void rand_sym(
	MKL_INT n, 
	double eigmin, 
	double eigmax, 
	double *Q,
	MKL_INT *iseed);

void rand_perturb(
	char trans, 
	MKL_INT n,  
	double maxnorm, 
	const double *A, 
	double *G, 
	MKL_INT *iseed);

void rand_orth(
	MKL_INT n, 
	MKL_INT m, 
	double *B,
	MKL_INT *iseed);

/* IO */
void print_vec(
	FILE *fp, 
	MKL_INT n, 
	const double *X);

void print_mat(
	FILE *fp, 
	char lo, 
	MKL_INT m, 
	MKL_INT n, 
	const double *A);	

/* SLICOT */
void MA02ED(
	const char *uplo, 
	const MKL_INT *n, 
	double *a, 
	const MKL_INT *lda);

void MB01RU(
	const char *uplo, 
	const char *trans, 
	const MKL_INT *m, 
	const MKL_INT *n, 
	const double *alpha, 
	const double *beta, 
	double *r, 
	const MKL_INT *ldr, 
	const double *a, 
	const MKL_INT *lda, 
	const double *x, 
	const MKL_INT *ldx, 
	double *dwork, 
	const MKL_INT *ldwork, 
	MKL_INT *info);

void SB03RD(
	const char *job, 
	const char *fact, 
	const char *trana, 
	const MKL_INT *n, 
	double *a, 
	const MKL_INT *lda, 
	double *u, 
	const MKL_INT *ldu, 
	double *c, 
	const MKL_INT *ldc, 
	double *scale, 
	double *sep, 
	double *ferr, 
	double *wr, 
	double *wi, 
	MKL_INT *iwork, 
	double *dwork, 
	const MKL_INT *ldwork, 
	MKL_INT *info);

void SG02CX(
	const char *jobe,
	const char *flag,
	const char *jobg, 
	const char *uplo, 
	const char *trans, 
	const MKL_INT *n, 
	const MKL_INT *m, 
	const double *e, 
	const MKL_INT *lde, 
	const double *r,
    const MKL_INT *ldr, 
    const double *s, 
    const MKL_INT *lds, 
    const double *g, 
    const MKL_INT *ldg, 
    double *alpha, 
    double *rnorm, 
    double *dwork,
    const MKL_INT *ldwork, 
    MKL_INT *iwarn, 
    MKL_INT *info);
#endif
