#include "mex.h"
#include "common.h"

/**
 * [A] = rand_mat(n, remin, remax)
 */

void mexFunction(
	int nlhs, 
	mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[])
{
	if (nrhs != 3)
    	mexErrMsgIdAndTxt("ControlSim:rand_mat:nrhs", "Three input arguments required.");

	if (nlhs != 1) 
		mexErrMsgIdAndTxt("ControlSim:rand_mat:nlhs", "One output argument required.");

	if (!mxIsInt64(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1 || mxGetScalar(prhs[0]) <= 0)
		mexErrMsgIdAndTxt("ControlSim:gen_care:notIntegerScalar", 
			"First input argument must be a positive scalar of type int64.");

	if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
    	mexErrMsgIdAndTxt("ControlSim:rand_mat:notDoubleScalar", 
    		"Second input argument must be a scalar of type double.");

	if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
    	mexErrMsgIdAndTxt("ControlSim:rand_mat:notDoubleScalar", 
    		"Third input argument must be a scalar of type double.");

	const MKL_INT n = (MKL_INT) mxGetScalar(prhs[0]);
	const double remin = mxGetScalar(prhs[1]);
	const double remax = mxGetScalar(prhs[2]);

	plhs[0] = mxCreateDoubleMatrix((mwSize) n, (mwSize) n, mxREAL);
	double *A = mxGetPr(plhs[0]);
	
	MKL_INT iseed[4];
	gen_seed(iseed);
	rand_mat(n, remin, remax, A, iseed);
}