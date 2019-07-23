#include "mex.h"
#include "common.h"

/**
 *	[X, nstep] = gen_care(A, G, gamma, B, Q, T, kmax);
 */	

void mexFunction(
	int nlhs, 
	mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[])
{
	if (nrhs != 7) 
    	mexErrMsgIdAndTxt("ControlSim:gen_care:nrhs", "Seven input arguments required.");

	if (nlhs != 2) 
		mexErrMsgIdAndTxt("ControlSim:gen_care:nlhs", "Two output arguments required.");

	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxGetM(prhs[0]) != mxGetN(prhs[0])) 
    	mexErrMsgIdAndTxt("ControlSim:gen_care:notSquareMatrixOfDouble", 
    		"First input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetM(prhs[1]) != mxGetN(prhs[1]))
    	mexErrMsgIdAndTxt("ControlSim:gen_care:notSquareMatrixOfDouble", 
    		"Second input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
    	mexErrMsgIdAndTxt("ControlSim:gen_care:notDoubleScalar", 
    		"Third input argument must be a scalar of type double.");

	if (!mxIsDouble(prhs[3]) || mxIsComplex(prhs[3]) || mxGetM(prhs[3]) != mxGetN(prhs[3]))
    	mexErrMsgIdAndTxt("ControlSim:gen_care:notSquareMatrixOfDouble", 
    		"Fourth input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[4]) || mxIsComplex(prhs[4]) || mxGetM(prhs[4]) != mxGetN(prhs[4])) 
    	mexErrMsgIdAndTxt("ControlSim:gen_care:notSquareMatrixOfDouble", 
    		"Fifth input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[5]) || mxIsComplex(prhs[5]) || mxGetM(prhs[5]) != mxGetN(prhs[5])) 
    	mexErrMsgIdAndTxt("ControlSim:gen_care:notSquareMatrixOfDouble", 
    		"Sixth input argument must be a square matrix of type double.");

	if (!mxIsInt64(prhs[6]) || mxGetNumberOfElements(prhs[6]) != 1)
		mexErrMsgIdAndTxt("ControlSim:gen_care:notIntegerScalar", 
			"Seventh input argument must be a scalar of type int64.");

 	if (mxGetM(prhs[0]) != mxGetM(prhs[1]) || mxGetM(prhs[1]) != mxGetM(prhs[3]) || mxGetM(prhs[3]) != mxGetM(prhs[4]))
		mexErrMsgIdAndTxt("ControlSim:gen_care:dimensionsDontMatch",
			"The dimensions of the first 4 input matrices must be the same.");

	if (mxGetM(prhs[0]) * (mxGetM(prhs[0]) + 1) / 2 != mxGetM(prhs[5]))
		mexErrMsgIdAndTxt("ControlSim:gen_care:dimensionsDontMatch",
			"The dimension of the fifth input matrix must be n*(n+1)/2, "
			"where n is the common dimension of the first 4 input matrices.");

	const MKL_INT n = mxGetM(prhs[0]);
	const double *A = mxGetPr(prhs[0]);
	const double *G = mxGetPr(prhs[1]);
	const double gamma = mxGetScalar(prhs[2]);
	const double *B = mxGetPr(prhs[3]);
	const double *Q = mxGetPr(prhs[4]);
	double *T = mxGetPr(prhs[5]);
	const MKL_INT kmax = (MKL_INT) mxGetScalar(prhs[6]);

	plhs[0] = mxCreateDoubleMatrix((mwSize) n, (mwSize) n, mxREAL);
	double *X = mxGetPr(plhs[0]);

	plhs[1] = mxCreateNumericMatrix(1, 1, mxINT64_CLASS, mxREAL);
	MKL_INT *nstep = (MKL_INT *) mxGetPr(plhs[1]);

	gen_care_solve_newton(kmax, n, gamma, A, G, B, Q, T, X, nstep); 
}