#include "common.h"

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
	double *restrict ngradJ)
{
	const MKL_INT n2 = n * n;
	MKL_INT nstep, info;
	const char uplo = 'L', trans = 'N';
	const CBLAS_TRANSPOSE cbtrans = CblasNoTrans;
	const double zero = 0.0, one = 1.0;
	double P[n * n], R[n * n], M[n *  n], W[n * n];
	
	// Solve A' * P + P * A + G' * P * G - gamma**2 P * B * P + Q = 0 for P.
	gen_care_solve_newton(kmax, n, gamma, A, G, B, Q, T, P, &nstep);
	// W := A - gamma**2 * B * P.
	LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', n, n, A, n, W, n);
	cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, n, n, -gamma, B, n, P, n, 1.0, W, n);
	// Solve W * R + R * W' + G * R * G' + L = 0 for R. 
	gen_lyap_solve_direct(n, -1.0, W, G, L, T, R);
	// M := P * R * P.
	MB01RU(&uplo, &trans, &n, &n, &zero, &one, M, &n, P, &n, R, &n, W, &n2, &info);
	MA02ED(&uplo, &n, M, &n);
	// W := M - B * M.
	LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', n, n, M, n, W, n);
	cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, n, n, -1.0, B, n, M, n, 1.0, W, n);
	// ngradJ := -gamma**2 * (W * B' + B * W') = -gamma**2 * [B, [B, M]], since B**2 = B = B', M = M', and thus:
	// [B, [B, M]] = B * (B * M - M * B) - (B * M - M * B) * B  = (M - B * M) * B + B * (M - M * B).
	cblas_dsyr2k(CblasColMajor, CblasLower, cbtrans, n, n, -gamma, W, n, B, n, 0.0, ngradJ, n);
}
