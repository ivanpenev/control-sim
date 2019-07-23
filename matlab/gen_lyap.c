#include "mex.h"
#include "common.h"

/**
 *	[X] = gen_lyap(A, G, Y, T);
 */	

void mexFunction(
	int nlhs, 
	mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[])
{
	if (nrhs != 4) 
    	mexErrMsgIdAndTxt("ControlSim:gen_lyap:nrhs", "Four input arguments required.");

	if (nlhs != 1) 
		mexErrMsgIdAndTxt("ControlSim:gen_lyap:nlhs", "One output argument required.");

	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxGetM(prhs[0]) != mxGetN(prhs[0])) 
    	mexErrMsgIdAndTxt("ControlSim:gen_lyap:notSquareMatrixOfDouble", 
    		"First input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1]) != mxGetN(prhs[1]))
    	mexErrMsgIdAndTxt("ControlSim:gen_lyap:notSquareMatrixOfDouble", 
    		"Second input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetM(prhs[2]) != mxGetN(prhs[2]))
    	mexErrMsgIdAndTxt("ControlSim:gen_lyap:notSquareMatrixOfDouble", 
    		"Third input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || mxGetM(prhs[3]) != mxGetN(prhs[3])) 
    	mexErrMsgIdAndTxt("ControlSim:gen_lyap:notSquareMatrixOfDouble", 
    		"Fourth input argument must be a square matrix of type double.");

 	if (mxGetM(prhs[0]) != mxGetM(prhs[1]) || mxGetM(prhs[1]) != mxGetM(prhs[2]))
		mexErrMsgIdAndTxt("ControlSim:gen_lyap:dimensionsDontMatch",
			"The dimensions of the first 3 input matrices must be the same.");

	if (mxGetM(prhs[0]) * (mxGetM(prhs[0]) + 1) / 2 != mxGetM(prhs[3]))
		mexErrMsgIdAndTxt("ControlSim:gen_lyap:dimensionsDontMatch",
			"The dimension of the fourth input matrix must be n*(n+1)/2, "
			"where n is the common dimension of the first 3 input matrices.");

	const char uplo = 'L';
	const MKL_INT n = mxGetM(prhs[0]);
	const double *A = mxGetPr(prhs[0]);
	const double *G = mxGetPr(prhs[1]);
	const double *Y = mxGetPr(prhs[2]);
	double *T = mxGetPr(prhs[3]);

	plhs[0] = mxCreateDoubleMatrix((mwSize) n, (mwSize) n, mxREAL);
	double *X = mxGetPr(plhs[0]);

	gen_lyap_solve_direct(n, -1.0, A, G, Y, T, X);
	MA02ED(&uplo, &n, X, &n); 
}