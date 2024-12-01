#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <cublas_v2.h>
#include <omp.h>
#include "../../lib/cuSolver.h"
#include "../../lib/Segment.h"
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
extern "C" {
#include "../../lib/RealMatrix.h"
using namespace std;



// LAPACK routine for solving linear system
void dgels_(const char * trans, const int * m, const int * n, const int * nrhs, double * A, const int * lda, double * B,  const int * ldb, double * work, int * lwork,int * info);


// Need to transpose manually as cusolver dgels does not seem to have implemented this yet
__global__ void transpose(double * input, double * output, int rows, int cols) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    if (r < rows && c < cols) {
        output[c*rows + r] = input[r*cols + c];
    }
}

void solveLinearSystem_CPU(RealMatrix A, RealMatrix b, ComplexMatrix C, ComplexMatrix D) {
    int n_int = C.rows;
    int n_ext = D.rows;
    char trans;
    int m, n, nrhs, lda, ldb, info, lwork;
    double work_query;
    double *work;

    trans = 'T';
    m = A.cols;
    n = A.rows;
    nrhs = 1; 
    lda = m;
    ldb = std::max(m, n);
    lwork = -1;

    dgels_(&trans, &m, &n, &nrhs, A.getHostPointer(), &lda, b.getHostPointer(), &ldb, &work_query, &lwork, &info);
    
    lwork = (int)work_query;
    work = (double*)malloc(lwork * sizeof(double));

    dgels_(&trans, &m, &n, &nrhs, A.getHostPointer(), &lda, b.getHostPointer(), &ldb, work, &lwork, &info);
    if (info != 0) {
        printf("An error occurred in solving: %d\n", info);
    }

    for (int i = 0; i < n_int; i++) {
        C.setHostRealValue(i,b.getHostValue(i));
        C.setHostImagValue(i,b.getHostValue(i + n_ext + n_int));
    }
    
    for (int i = 0; i < n_ext; i++) {
        D.setHostRealValue(i,b.getHostValue(i + n_int));
        D.setHostImagValue(i,b.getHostValue(i + n_ext + 2*n_int));
    }
}

__global__ void moveSolution(double * x, double * sol, int shift, int n) {
    // Moves solution from x to sol with shift
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        sol[i] = x[i + shift];
    }

}

void solveLinearSystem_GPU(RealMatrix A, RealMatrix b, ComplexMatrix C, ComplexMatrix D, double * A_T_d, double * x_d, cusolverDnHandle_t handle) {
    double start = omp_get_wtime();
  
    // Define variables for solving linear system
    int m = A.rows;
    int n = A.cols;
    int nrhs = 1; 
    int lda = m;
    int ldb = m;
    int ldx = n;

    // Move to device and transpose
    void * work_d = NULL;
    
    // Blocks and threads
    dim3 dimBlock(32,32);
    dim3 dimGrid((A.rows + dimBlock.x - 1)/dimBlock.x, (A.cols + dimBlock.y - 1)/dimBlock.y);
    transpose<<< dimGrid, dimBlock>>>(A.getDevicePointer(), A_T_d, A.rows, A.cols);
    cudaDeviceSynchronize();

    size_t lwork_bytes = 0;

    cusolverStatus_t status = cusolverDnDDgels_bufferSize(handle, m, n, nrhs, A_T_d, lda, b.getDevicePointer(), ldb, x_d, ldx, work_d, &lwork_bytes);
    cudaDeviceSynchronize();
    cudaMalloc((void **) &work_d, lwork_bytes);
    int niters;
    int *info_d;
    cudaMalloc((void**)&info_d, sizeof(int));
    double start_inner = omp_get_wtime();
    status = cusolverDnDDgels(handle, m, n, nrhs, A_T_d, lda, b.getDevicePointer(), ldb, x_d, ldx, work_d, lwork_bytes, &niters, info_d);
    cudaDeviceSynchronize();
    double end_inner = omp_get_wtime();
    //printf("time = %f\n",end-start);
    int info_h = 0;
    cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);
    //throw std::runtime_error("There was a problem with solving the system!");
    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("There was a problem with solving the system!\n");
    }
   

    if (info_h == 0) {
        //std::cout << "Solution was successful.\n";
    } else {
        std::cout << "Solution failed with info = " << info_h << std::endl;
    }

    int n_int = C.rows;
    int n_ext = D.rows;

    // Blocks and threads
    dim3 dimBlock2(256);
    dim3 dimGrid2((n_int + dimBlock.x - 1)/dimBlock.x);
    int shift = 0;
    moveSolution<<<dimGrid2, dimBlock2>>>(x_d, C.getDeviceRealPointer(), shift, n_int);
    shift = n_ext + n_int;
    moveSolution<<<dimGrid2, dimBlock2>>>(x_d, C.getDeviceImagPointer(), shift, n_int);
    dimGrid2.x = (n_ext + dimBlock2.x - 1)/dimBlock2.x;
    shift = n_int;
    moveSolution<<<dimGrid2, dimBlock2>>>(x_d, D.getDeviceRealPointer(), shift, n_ext);
    shift = n_ext + 2*n_int;
    moveSolution<<<dimGrid2, dimBlock2>>>(x_d, D.getDeviceImagPointer(), shift, n_ext);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(work_d);
    cudaFree(info_d);

    double end = omp_get_wtime();
    //printf("Linear solve part: %f\n",(end_inner-start_inner)/(end-start));


}

void Segment::solveLinearSystem() {

    char trans;
    int m, n, nrhs, lda, ldb, info, lwork;
    double work_query;
    double *work;

    if (deviceComputation) {
        solveLinearSystem_GPU(A, b, C, D, A_T_d, x_d, handle);
    }
    else {
        solveLinearSystem_CPU(A, b, C, D);
    }
     
}


}