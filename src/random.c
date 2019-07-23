#include "common.H"

#define MAX_SEED 4096
#define UNI_RAND (((double)rand()) / RAND_MAX)
#define RAND_INT(n) (rand() / (RAND_MAX / (n) + 1))

void gen_seed(MKL_INT iseed[static 4])
{
	for (MKL_INT i = 0; i < 4; ++i) iseed[i] = RAND_INT(MAX_SEED);
	iseed[3] |= 1;
}

void rand_eigen(
	MKL_INT n, 
	MKL_INT nr, 
	double remin, 
	double remax, 
	double *x, 
	char *type,
	MKL_INT *iseed)
{
	const double scale = remax - remin, shift = remin;

	for (MKL_INT i = 0; i < nr; ++i) {
		x[i] = scale * UNI_RAND + shift;
	}
	if (nr == n) {
		type[0] = ' ';
		return;
	}
	memset(type, 'R', nr);	
	for (MKL_INT i = nr; i < n; i += 2) {
		x[i] = scale * UNI_RAND + shift;
		type[i] = 'R';
		x[i + 1] = UNI_RAND;
		type[i + 1] = 'I';
	}
}

void rand_mat(
	MKL_INT n, 
	double remin, 
	double remax, 
	double *A,
	MKL_INT *iseed)
{
	MKL_INT nr, mode = 0, modes = 5, kl = n - 1, ku = n - 1, info;
	double cond, dmax, conds = 2.0, anorm = -1.0; 
	char dist = 'S', rsign = 'F', upper = 'F', sim = 'T';
	char ei[n];
	double d[n], ds[n], work[3 * n];

	nr = RAND_INT(n);
	// Ensure that (n - nr) is even.
	if (n % 2 != nr % 2) ++nr;
	rand_eigen(n, nr, remin, remax, d, ei, iseed);
	// Generate a random matrix A of the form X * D * X**(-1), where D is a block-diagonal matrix
	// with eigenvalues given by d, and X is an invertible matrix with singular values between 0.5 and 1.0.  
	LAPACKE_dlaset(LAPACK_COL_MAJOR, 'F', n, n, 0.0, 0.0, A, n);
	dlatme(&n, &dist, iseed, d, &mode, &cond, &dmax, ei, &rsign, &upper, 
		&sim, ds, &modes, &conds, &kl, &ku, &anorm, A, &n, work, &info);
}

void rand_sym(
	MKL_INT n, 
	double eigmin, 
	double eigmax, 
	double *Q,
	MKL_INT *iseed)
{
	double d[n];
	char ei[n];

	rand_eigen(n, n, eigmin, eigmax, d, ei, iseed);
	LAPACKE_dlagsy(LAPACK_COL_MAJOR, n, n - 1, d, Q, n, iseed);
}

void rand_perturb(
	char trans, 
	MKL_INT n, 
	double maxnorm, 
	const double *A, 
	double *G,
	MKL_INT *iseed)
{
	double Gt[n * n], GG[n * n], H[n * n], w[n];
	
	rand_mat(n, -1.0, 1.0, G, iseed);
	mkl_domatcopy('C', 'T', n, n, 1.0, G, n, Gt, n);
	if (trans == 'T')
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0, Gt, n, G, n, 0.0, GG, n);
	else 
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, -1.0, G, n, Gt, n, 0.0, GG, n);	
	lyap_solve(trans, n, A, GG, H);
	LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'L', n, H, n, w);
	if (w[n - 1] > maxnorm) 
		LAPACKE_dlascl(LAPACK_COL_MAJOR, 'G', n - 1, n - 1, sqrt(w[n - 1]), sqrt(maxnorm), n, n, G, n);		
}

void rand_orth(
	MKL_INT n, 
	MKL_INT m, 
	double *B,
	MKL_INT *iseed)
{	
	double d[m];
	const double one = 1.0;

	cblas_dcopy(m, &one, 0, d, 1);
	LAPACKE_dlagge(LAPACK_COL_MAJOR, n, m, n - 1, m - 1, d, B, n, iseed);
}
