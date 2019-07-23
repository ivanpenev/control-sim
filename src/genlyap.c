#include "common.h"

void lyap_solve(
	char trans, 
	MKL_INT n, 
	const double *restrict A, 
	const double *restrict Y, 
	double *X)
{
	const MKL_INT ldwork = (n >= 3? n: 3) * n;
	MKL_INT iwork[n * n];
	MKL_INT info;
	const char job = 'X', fact = 'N';
	double scale, sep, ferr;
	double Acopy[n * n], U[n * n], wr[n], wi[n], dwork[ldwork];
	
	LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', n, n, A, n, Acopy, n);
	LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', n, n, Y, n, X, n);
	SB03RD(&job, &fact, &trans, &n, Acopy, &n, U, &n, X, &n, &scale, 
		&sep, &ferr, wr, wi, iwork, dwork, &ldwork, &info);
	cblas_dscal(n * n, 1.0/scale, X, 1);
}

void gen_lyap_mat(
	MKL_INT n, 
	const double *restrict A, 
	const double *restrict G, 
	double *restrict T)
{
	const MKL_INT p = (n + 1) / 2, q = n - p, r = (n + 1) % 2, m = n + r, mp = m * p;
	MKL_INT i, j, k, l, ik, jl;
	bool below_diag;

	#pragma omp parallel for
	for (jl = 0; jl < mp; ++jl) {
		below_diag = (jl % m - r >= jl / m);
		l = below_diag? (jl % m - r): (jl / m + q);
		j = below_diag? (jl / m): (jl % m + p);
		#define INNER_LOOPS \
			for (i = 0; i < p; ++i) { \
				_Pragma ("omp simd") \
				for (k = i; k < n; ++k) { \
					ik = (k + r) + m * i; \
					LOOP_BODY \
				} \
			} \
			for (k = p; k < n; ++k) { \
				_Pragma ("omp simd") \
				for (i = p; i <= k; ++i) { \
					ik = (i - p) + m * (k - q); \
					LOOP_BODY \
				} \
			}

		if (j != l) {
			#define LOOP_BODY \
				ELT(T, mp, ik, jl) = ELT(G, n, i, j) * ELT(G, n, k, l) + \
				                     ELT(G, n, k, j) * ELT(G, n, i, l); \
				if (k == l) ELT(T, mp, ik, jl) += ELT(A, n, i, j); \
				if (i == j) ELT(T, mp, ik, jl) += ELT(A, n, k, l); \
				if (i == l) ELT(T, mp, ik, jl) += ELT(A, n, k, j); \
				if (k == j) ELT(T, mp, ik, jl) += ELT(A, n, i, l);	

			INNER_LOOPS
			#undef LOOP_BODY
		}
		else {
			#define LOOP_BODY \
				ELT(T, mp, ik, jl) = ELT(G, n, i, j) * ELT(G, n, k, j); \
				if (k == j) ELT(T, mp, ik, jl) += ELT(A, n, i, j); \
				if (i == j) ELT(T, mp, ik, jl) += ELT(A, n, k, j); 

			INNER_LOOPS
			#undef LOOP_BODY
		}
		#undef INNER_LOOPS		
	}
}

void gen_lyap_solve_direct( 
	MKL_INT n, 
	double alpha,
	const double *restrict A, 
	const double *restrict G, 
	const double *restrict Y,
	double *restrict T,
	double *restrict X)
{
	const MKL_INT mp = n * (n + 1) / 2;
	MKL_INT ipiv[mp];
	double Yp[mp];

	// Store the matrix of (L_A + Pi_G) : Sym(n) --> Sym(n) in T.
	gen_lyap_mat(n, A, G, T);
	// Pack the RHS in RFP format and scale it.
	LAPACKE_dtrttf(LAPACK_COL_MAJOR, 'N', 'L', n, Y, n, Yp);
	cblas_dscal(mp, alpha, Yp, 1);
	// Solve T * Xp = Yp for Xp storing the result in Yp.
	LAPACKE_dgesv(LAPACK_COL_MAJOR, mp, 1, T, mp, ipiv, Yp, mp);
	// Unpack the solution and store it in lo.tri.(X).
	LAPACKE_dtfttr(LAPACK_COL_MAJOR, 'N', 'L', n, Yp, X, n);
}

