#include "common.h"

#define FORMAT "%e"

void print_vec(
	FILE *fp, 
	MKL_INT n, 
	const double *x)
{
	for (MKL_INT i = 0; i < n; ++i) {
		fprintf(fp, FORMAT, x[i]);
		if (i < n - 1) fprintf(fp, "\t");
	}
	fprintf(fp, "\n");
}

void print_mat(
	FILE *fp, 
	char lo, 
	MKL_INT m, 
	MKL_INT n, 
	const double *A)
{
	for (MKL_INT i = 0; i < m; ++i) {
		MKL_INT jend = (lo == 'L'? i + 1: n);
		for (MKL_INT j = 0; j < jend; ++j) {
			fprintf(fp, FORMAT, ELT(A, m, i, j));
			if (j < jend - 1) fprintf(fp, "\t");
		}
		fprintf(fp, "\n");
	}
}
