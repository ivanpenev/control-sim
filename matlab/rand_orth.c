#include "mex.h"
#include "common.h"

/**
 * [B] = rand_sym(n, m)
 */

void mexFunction(
	int nlhs, 
	mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[])
{
	if (nrhs != 2)
    	mexErrMsgIdAndTxt("ControlSim:rand_orth:nrhs", "Two input arguments required.");

	if (nlhs != 1) 
		mexErrMsgIdAndTxt("ControlSim:rand_orth:nlhs", "One output argument required.");

	if (!mxIsInt64(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1 || mxGetScalar(prhs[0]) <= 0)
		mexErrMsgIdAndTxt("ControlSim:rand_orth:notIntegerScalar", 
			"First input argument must be a positive scalar of type int64.");

	if (!mxIsInt64(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1 || mxGetScalar(prhs[1]) <= 0 ||
		mxGetScalar(prhs[1]) > mxGetScalar(prhs[0]))
		mexErrMsgIdAndTxt("ControlSim:rand_orth:notIntegerScalar", 
			"Second input argument must be a positive scalar of type int64, not greater than the first input argumant.");

	const MKL_INT n = (MKL_INT) mxGetScalar(prhs[0]);
	const MKL_INT m = (MKL_INT) mxGetScalar(prhs[1]);
	
	plhs[0] = mxCreateDoubleMatrix((mwSize) n, (mwSize) m, mxREAL);
	double *B = mxGetPr(plhs[0]);
	
	MKL_INT iseed[4];
	gen_seed(iseed);
	rand_orth(n, m, B, iseed);
}