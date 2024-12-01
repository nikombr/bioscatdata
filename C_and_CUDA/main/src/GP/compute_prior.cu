#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
extern "C" {
#include "../../lib/GP/GaussianProcess.h"

__global__ void copyVector(double * input, double * output, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        output[i] = input[i];
    }
}

double GaussianProcess::compute_prior(double * f_d) {

    double val = 0;

    // cuBLAS handle creation
    cublasHandle_t handle;
    cublasStatus_t status;
    status = cublasCreate(&handle);

    double * x_d;
    cudaMalloc((void**) &x_d, n * sizeof(double));
    dim3 dimBlock(32);
    dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);
    copyVector<<<dimGrid, dimBlock>>>(f_d, x_d, n);

    /*double alpha = 1.0;
    double beta = 0.0;
    status = cublasDgemv(handle, trans, n, n, &alpha, M_inv_log, n, f_d, 1, &beta, x_d, 1);*/
    status = cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, M_inv_log, n, x_d, 1);
    cudaDeviceSynchronize();
    status = cublasDdot(handle, n, f_d, 1, x_d, 1, &val);
    cudaDeviceSynchronize();


    /*printf("val = %f\n",val+n/2*log(2*M_PI));
    printf("logDetermingn = %f\n",logDeterminant);
    printf("hmm %f\n",n/2*log(2*M_PI));
    double prior = -0.5*val - 0.5*logDeterminant - 0.5*n*log(2*M_PI);
    printf("prior = %f\n",prior);*/
    return -val;
}
}