#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusolverDn.h>
extern "C" {
#include "../../lib/GP/GaussianProcess.h"

// Need to transpose manually as cusolver does not seem to have implemented this yet
__global__ void transposeMatrix(double ** input, double ** output, int n) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    if (r < n && c < n) {
        output[c][r] = input[r][c];
    }
}

__global__ void copyMatrix(double ** input, double ** output, int n) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    if (r < n && c < n) {
        output[r][c] = input[r][c];
    }
}


void GaussianProcess::compute_inverse() {


    if (device) {

        cusolverStatus_t status;

        // Move the cholesky factorized covariance matrix to avoid overwrites
        dim3 dimBlock(32,32);
        dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x, (n + dimBlock.y - 1)/dimBlock.y);
        copyMatrix<<<dimGrid, dimBlock>>>(M_d, M_inv_d, n);
        cudaDeviceSynchronize();

        // Get cuSolver handle
        cusolverDnHandle_t handle;
        cusolverDnCreate(&handle);

        cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

        // Workspace query
        int work_size = 0;
        status =  cusolverDnDpotri_bufferSize(handle, uplo, n, M_inv_log, n, &work_size);

        // Allocate workspace
        double * workspace_d;
        cudaMalloc((void**) &workspace_d, work_size * sizeof(double));
        int * info_d;
        cudaMalloc((void**)&info_d, sizeof(int));

        // Get inverse
        status = cusolverDnDpotri(handle, uplo, n, M_inv_log, n, workspace_d, work_size, info_d);
        cudaDeviceSynchronize();
        
        // Check if successful
        int info_h = 0;
        cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);
        if (info_h != 0) {
            printf("Computing inverse failed with info = %d\n", info_h);
        }
        

        cudaFree(workspace_d);
        cudaFree(info_d);
        cusolverDnDestroy(handle);
        cudaDeviceSynchronize();


    }
    else {
        printf("Error: Not implemented on host!\n"); // Fine as we probably won't use it on the host
    }


}

}