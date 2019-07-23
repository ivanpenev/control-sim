#include "mex.h"
#include "common.h"

/**
 *	[ngradJ] = mincost_ngrad(s, B, gamma, A, G, Q, L, T, kmax);
 */	

void mexFunction(
	int nlhs, 
	mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[])
{
	if (nrhs != 9) 
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:nrhs", "Nine input arguments required.");

	if (nlhs != 1) 
		mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:nlhs", "One output argument required.");

	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxGetNumberOfElements(prhs[2]) != 1)
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notDoubleScalar", 
    		"First input argument must be a scalar of type double.");

	if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetN(prhs[1]) != 1) 
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notSquareMatrixOfDouble", 
    		"Second input argument must be a column vector of type double.");

	if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notDoubleScalar", 
    		"Third input argument must be a scalar of type double.");

	if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || mxGetM(prhs[3]) != mxGetN(prhs[3]))
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notTallMatrixOfDouble", 
    		"Fourth input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxGetM(prhs[4]) != mxGetN(prhs[4])) 
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notSquareMatrixOfDouble", 
    		"Fifth input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxGetM(prhs[5]) != mxGetN(prhs[5])) 
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notSquareMatrixOfDouble", 
    		"Sixth input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[6]) || mxIsComplex(prhs[6]) || mxGetM(prhs[6]) != mxGetN(prhs[6])) 
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notSquareMatrixOfDouble", 
    		"Seventh input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[7]) || mxIsComplex(prhs[7]) || mxGetM(prhs[7]) != mxGetN(prhs[7]))
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notSquareMatrixOfDouble", 
    		"Eighth input argument must be a square matrix of type double.");

	if (!mxIsInt64(prhs[8]) || mxGetNumberOfElements(prhs[8]) != 1)
		mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notIntegerScalar", 
			"Ninth input argument must be a scalar of type int64.");

 	if (mxGetM(prhs[3]) != mxGetM(prhs[4]) || mxGetM(prhs[4]) != mxGetM(prhs[5]) || 
 		mxGetM(prhs[5]) != mxGetM(prhs[6]))
		mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:dimensionsDontMatch",
			"The dimensions of the first 4 input square matrices must be the same.");

	if (mxGetM(prhs[1]) != mxGetM(prhs[3]) * mxGetM(prhs[3]))
		mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:dimensionsDontMatch",
			"The dimension of the input row vector must be n*n, "
			"where n is the common dimension of the first 4 input square matrices.");

	if (mxGetM(prhs[3]) * (mxGetM(prhs[3]) + 1) / 2 != mxGetM(prhs[7]))
		mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:dimensionsDontMatch",
			"The dimension of the fifth input square matrix must be n*(n+1)/2, "
			"where n is the common dimension of the first 4 input square matrices.");

	const char uplo = 'L';
	const double *B = mxGetPr(prhs[1]);
	const double gamma = mxGetScalar(prhs[2]);
	const MKL_INT n = mxGetM(prhs[3]);
	const double *A = mxGetPr(prhs[3]);
	const double *G = mxGetPr(prhs[4]);
	const double *Q = mxGetPr(prhs[5]);
	const double *L = mxGetPr(prhs[6]);
	double *T = mxGetPr(prhs[7]);
	const MKL_INT kmax = (MKL_INT) mxGetScalar(prhs[8]);

	plhs[0] = mxCreateDoubleMatrix((mwSize) (n * n), 1, mxREAL);
	double *ngradJ = mxGetPr(plhs[0]);
	MKL_INT nstep;

	min_cost_ngrad(kmax, n, gamma, A, G, B, Q, L, T, ngradJ, &nstep);
	MA02ED(&uplo, &n, ngradJ, &n);
}