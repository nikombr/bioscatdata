

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cusolverDn.h>
extern "C" {
#include "../lib/GaussianProcess.h"

// LAPACK routine for Cholesky factorization
void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);

void GaussianProcess::cholesky() {

    if (device) {

        // Create cuSOLVER handle
        cusolverDnHandle_t cusolverH;
        cusolverDnCreate(&cusolverH);

        // Get workspace size
        int workspace_size = 0;
        cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_UPPER, n, M_log, n, &workspace_size);

        // Allocate workspace
        double *workspace;
        cudaMalloc((void**)&workspace, workspace_size * sizeof(double));

        // Allocate info variable
        int *info_d,*info_h;
        cudaMalloc((void**)&info_d, sizeof(int));
        cudaMallocHost((void**)&info_h, sizeof(int));

        // Cholesky factorization on device
        cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_UPPER, n, M_log, n, workspace, workspace_size, info_d);
        cudaDeviceSynchronize();

        cudaMemcpy(info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);

        if (*info_h == 0) {
            printf("Cholesky factorization successful.\n\n");
        } else if (*info_h > 0) {
            printf("Matrix is not positive definite at leading minor %d.\n\n", *info_h);
        } else {
            printf("Error with argument %d.\n\n", -*info_h);
        }

    }
    else {
        char uplo = 'U';
        int N = n*n;
        int info;
        dpotrf_(&uplo, &n, *M_h, &n,&info);

        if (info == 0) {
            printf("Cholesky factorization successful.\n\n");
        } else if (info > 0) {
            printf("The leading minor of order %d is not positive definite.\n\n", info);
        } else {
            printf("Illegal value in LAPACKE_dpotrf arguments.\n\n");
        }
  
        
    }
    
}

}