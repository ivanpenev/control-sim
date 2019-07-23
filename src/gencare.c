#include "common.h"

// Y := Ric(X) = Q + op(G) * X * op(G)' + op(A) * X + X * op(A)' - gamma * X * B * X
void gen_care_eval(
	char trans,
	MKL_INT n,
	double gamma, 
	const double *restrict A,
	const double *restrict G,
	const double *restrict B,
	const double *restrict Q,
	const double *restrict X,
	double *restrict Y)
{
	const MKL_INT n2 = n * n;
	MKL_INT info;
	const char uplo = 'L', notrans = 'N';
	const CBLAS_SIDE side = (trans == 'T'? CblasLeft: CblasRight);
	const CBLAS_TRANSPOSE cbtrans = (trans == 'T'? CblasTrans: CblasNoTrans); 
	const double one = 1.0;
	double W[n * n];

	// lo.tri.(Y) := lo.tri.(Q + op(G) * X * op(G)').
	LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'L', n, n, Q, n, Y, n);
	MB01RU(&uplo, &trans, &n, &n, &one, &one, Y, &n, G, &n, X, &n, W, &n2, &info);
	// W := A - (1/2) * gamma * X * B (if trans = 'N'), or W :=  A - (1/2) * gamma * B * X (if trans == 'T').
	LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', n, n, A, n, W, n);
	cblas_dsymm(CblasColMajor, side, CblasLower, n, n, -0.5 * gamma, B, n, X, n, 1.0, W, n);
	// lo.tri.(Y) := lo.tri.(Y + op(W) * X' + X * op(W)'.
	cblas_dsyr2k(CblasColMajor, CblasLower, cbtrans, n, n, 1.0, W, n, X, n, 1.0, Y, n);
}


// Solve A' * X + X * A + G' * X * G - gamma * X * B * X + Q = 0 for X
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
	MKL_INT *nstep)
{
	const MKL_INT n2 = n * n, one = 1, ldwork = (n2 >= 51? 2 * n2: n2 + 51);
	MKL_INT k, iwarn, info;
	const char uplo = 'L', jobe = 'I', flag = 'M', jobg = 'G', trans = 'N';
	// Relative machine precision (epsilon) = 2**(-52).
	const double eps = 2.220446049250313e-16;
	const double anorm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, A, n);
	const double gnorm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', n, n, G, n);
	const double bnorm = LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, B, n);
	const double qnorm = LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, Q, n);
	// Compute the numerical tolerance, in terms of the norms of the coefficients, following [Benner-Sima].
	const double tol = fmin(eps * sqrt(n) * (2 * anorm + gnorm * gnorm + gamma * bnorm + qnorm), sqrt(eps));
	const double tols = 0.9;
	double xnorm, dxnorm, resnorm[2] = {DBL_MAX}, alpha, rnorm, dummy;
	double Y[n * n], At[n * n], Gt[n * n], dX[n * n], gammaB[n * n], dwork[ldwork];

	mkl_domatcopy('C', 'T', n, n, 1.0, G, n, Gt, n);
	mkl_domatcopy('C', 'N', n, n, gamma, B, n, gammaB, n);
	LAPACKE_dlaset(LAPACK_COL_MAJOR, 'A', n, n, 0.0, 0.0, X, n);
	// Prevent garbage values in the upper triangle of Y from crashing LAPACKE routines due to NaN checks.
	LAPACKE_dlaset(LAPACK_COL_MAJOR, 'U', n, n, 0.0, 0.0, Y, n);
	
	for (k = 0; k < kmax; ++k) {
		// Compute the Frobenius norm of X_k.
		xnorm = LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, X, n);
		// lo.tri.(Y_k) := lo.tri.(Ric(X_k)), where Ric(X) =  Q + G' * X * G + A' * X + X * A - gamma * X * B * X.
		gen_care_eval('T', n, gamma, A, G, B, Q, X, Y);
		// Save the previously computed norm of the residual.
		resnorm[1] = resnorm[0];
		// Compute the normalized norm of the residual, Y_k.
		resnorm[0] = LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, Y, n) / fmax(1.0, xnorm);
		// Stop if the solution has been computed to the pre-set precision.
		if (resnorm[0] <= tol) break;
		// A_k' := A' - gamma * X_k * B.
		mkl_domatcopy('C', 'T', n, n, 1.0, A, n, At, n);
		cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, n, n, -gamma, X, n, B, n, 1.0, At, n);
		// Solve A_k' * dX_k + dX_k * A_k + G' * dX_k * G = -Y_k for dX_k.
		gen_lyap_solve_direct(n, -1.0, At, Gt, Y, T, dX);
		MA02ED(&uplo, &n, dX, &n);
		// Find alpha = alpha_k in [0, 2] that minimizes the Frobenius norm of the residual 
		// Ric(X_k + alpha * dX_k) = (1 - alpha) Y_k - alpha**2 * dX_k * (gamma * B) * dX_k.
		// The corresponding minimum value of the Frobenius norm of the residual is stored in rnorm.
		SG02CX(&jobe, &flag, &jobg, &uplo, &trans, &n, &n, &dummy, &one, Y, &n, dX, &n, gammaB, &n, 
 			&alpha, &rnorm, dwork, &ldwork, &iwarn, &info);
		// Check if a standard Newton step is preferable, following [Benenr-Sima]. 
		if ((n > 1 && k <= 10 && alpha < 0.5 && 
			resnorm[0] > pow(eps, 0.25) && resnorm[0] < 1.0 && rnorm <= 10.0) || 
			rnorm > tols * resnorm[1])
		{
			alpha = 1.0;
			resnorm[0] = 0.0;
		}
		// Stop if updating X_k cannot improve the solution, [Benner-Sima].
		dxnorm = LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, dX, n);
		if (alpha * dxnorm <= eps * xnorm) {
			*nstep = -k - 1;
			return;
		}
		// X_{k+1} := X_k + alpha_k * dX_k.
		cblas_daxpy(n * n, alpha, dX, 1, X, 1);
	}
	*nstep = k;
}
