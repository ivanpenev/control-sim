#include "common.h"

int main(int argc, char *argv[])
{
	const MKL_INT n = (argc > 1? atoi(argv[1]): 10), m = n/2, mp = n * (n + 1) / 2;
	const double gamma = 0.5;
	double A[n * n], G[n * n], B[n * m], BtB[m * m], BBt[n * n], Q[n * n], X[n * n], Y[n * n];
	double *T = malloc(mp * mp * sizeof(double));
	if (T == NULL) return -1;
	MKL_INT iseed[4], nstep;
	const char trans = 'T', uplo = 'L';
	FILE *fp;

	srand(time(NULL));
	gen_seed(iseed);
	rand_mat(n, -5.0, -1.0, A, iseed);
	rand_perturb(trans, n, 0.75, A, G, iseed);
	rand_sym(n, 1.0, 4.0, Q, iseed);
	rand_orth(n, m, B, iseed);
	cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, m, n, 1.0, B, n, 0.0, BtB, m);
	MA02ED(&uplo, &m, BtB, &m);
	cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, m, 1.0, B, n, 0.0, BBt, n);
	MA02ED(&uplo, &n, BBt, &n);

	gen_care_solve_newton(20, n, gamma, A, G, BBt, Q, T, X, &nstep);
	gen_care_eval('T', n, gamma, A, G, BBt, Q, X, Y);
  	fp = fopen("gencare_out.txt", "w");
  	if (fp == NULL) {
  		free(T);
  		return -1;
  	}
	fprintf(fp, "A:\n");
	print_mat(fp, 'A', n, n, A);
	fprintf(fp, "\nG:\n");
	print_mat(fp, 'A', n, n, G);
	fprintf(fp, "\nB:\n");
	print_mat(fp, 'A', n, m, B);
	fprintf(fp, "\nB'B:\n");
	print_mat(fp, 'A', m, m, BtB);
	fprintf(fp, "\nBB':\n");
	print_mat(fp, 'A', n, n, BBt);
	fprintf(fp, "\nQ:\n");
	print_mat(fp, 'A', n, n, Q);
	fprintf(fp, "\nX:\n");
	print_mat(fp, 'A', n, n, X);
	fprintf(fp, "\nY = A'X + XA + G'XG - gamma XBB'X + Q:\n");
	print_mat(fp, 'L', n, n, Y);
	fprintf(fp, "\n|Y| = %e\n", LAPACKE_dlansy(LAPACK_COL_MAJOR, 'F', 'L', n, Y, n));
	fprintf(fp, "\n%d Newton steps", nstep);
	fclose(fp);
	free(T);
	return 0;
}