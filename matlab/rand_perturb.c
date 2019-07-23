#include "mex.h"
#include "common.h"

/**
 * [G] = rand_perturb(A, maxnorm)
 */

void mexFunction(
	int nlhs, 
	mxArray *plhs[],
    int nrhs, 
    const mxArray *prhs[])
{
	if (nrhs != 2)
    	mexErrMsgIdAndTxt("ControlSim:rand_perturb:nrhs", "Two input arguments required.");

	if (nlhs != 1) 
		mexErrMsgIdAndTxt("ControlSim:rand_perturb:nlhs", "One output argument required.");

	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxGetM(prhs[0]) != mxGetN(prhs[0])) 
    	mexErrMsgIdAndTxt("ControlSim:mincost_ngrad:notSquareMatrixOfDouble", 
    		"First input argument must be a square matrix of type double.");

	if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
    	mexErrMsgIdAndTxt("ControlSim:rand_mat:notDoubleScalar", 
    		"Second input argument must be a scalar of type double.");

	const MKL_INT n = mxGetM(prhs[0]);
	const double *A = mxGetPr(prhs[0]);
	const double maxnorm = mxGetScalar(prhs[1]);

	plhs[0] = mxCreateDoubleMatrix((mwSize) n, (mwSize) n, mxREAL);
	double *G = mxGetPr(plhs[0]);
	
	MKL_INT iseed[4];
	gen_seed(iseed);
	rand_perturb('T', n, maxnorm, A, G, iseed);
}