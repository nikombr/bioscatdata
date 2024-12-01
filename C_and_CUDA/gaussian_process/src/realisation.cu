#include <stdlib.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>

extern "C" {
#include "../lib/GaussianProcess.h"
#include <cblas.h>
#include <ctime>


void GaussianProcess::generate_random_vector() {

    double pi = 3.14159265358979323846;
    double U1, U2;

    for (int i = 0; i < n; i+=2) {
        U1 = ((double) rand())/((double) RAND_MAX);
        U2 = ((double) rand())/((double) RAND_MAX);
        p_h[i]   = sqrt(-2*log(U1))*cos(2*pi*U2);
        p_h[i+1] = sqrt(-2*log(U1))*sin(2*pi*U2);

    }

    if (n < 21) {
        printf("\n");
        for (int k = 0; k < n; k++) {
            if (k != n-1) printf("%.8f, ",p_h[k]);
            else printf("%.8f",p_h[k]);
            
        }
        printf("\n");
    }

    if (device) {
        // Send to device
        cudaMemcpy(p_d, p_h, n * sizeof(double), cudaMemcpyHostToDevice);
    }


}

void GaussianProcess::realisation() {

    generate_random_vector();

    if (device) {

        // cuBLAS handle creation
        cublasHandle_t handle;
        cublasStatus_t status;
        status = cublasCreate(&handle);

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS initialization failed %d\n",status);
            return;
        }
        
        status = cublasDtrmv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, M_log, n, p_d, 1);
        
        // Check if cublasDtrmv was successful
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cublasDtrmv failed with error code: %d\n", status);
        }

        // Send to host
        cudaMemcpy(p_h, p_d, n * sizeof(double), cudaMemcpyDeviceToHost);

        // Destroy cuBLAS handle
        status = cublasDestroy(handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("CUBLAS destruction failed\n");
            return;
        }
    }
    else {
        cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, n, *M_h, n, p_h, 1);
    }

}


}