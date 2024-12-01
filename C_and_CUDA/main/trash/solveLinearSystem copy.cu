#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <cublas_v2.h>
#include <omp.h>
#include <cusolverDn.h>
extern "C" {
#include "../../lib/Segment.h"
#include "../../lib/RealMatrix.h"
using namespace std;

// LAPACK routine for solving linear system
void dgels_(const char * trans, const int * m, const int * n, const int * nrhs, double * A, const int * lda, double * B,  const int * ldb, double * work, int * lwork,int * info);


// Need to transpose manually as cublas dgels does not seem to have implemented this yet
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

void solveLinearSystem_GPU(RealMatrix A, RealMatrix b, ComplexMatrix C, ComplexMatrix D) {
    // Define variables for solving linear system
    int m = A.rows;
    int n = A.cols;
    int nrhs = 1; 
    int lda = m;
    int ldb = m;
    int ldx = n;

    // Move to device and transpose
    double * A_d, *A_T_d, *b_d, *x_h, *x_d;
    void * work_d = NULL;

    cudaMalloc((void **) &A_T_d,     A.rows * A.cols * sizeof(double));
    //cudaMallocHost((void **) &x_h,     A.cols * sizeof(double));
    cudaMalloc((void **) &x_d,     A.cols * sizeof(double));
    
    // Blocks and threads
    dim3 dimBlock(32,32);
    dim3 dimGrid((A.rows + dimBlock.x - 1)/dimBlock.x, (A.cols + dimBlock.y - 1)/dimBlock.y);
    transpose<<< dimGrid, dimBlock>>>(A.getDevicePointer(), A_T_d, A.rows, A.cols);
    cudaDeviceSynchronize();
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    size_t lwork_bytes = 0;

    cusolverStatus_t status = cusolverDnDDgels_bufferSize(handle, m, n, nrhs, A_T_d, lda, b.getDevicePointer(), ldb, x_d, ldx, work_d, &lwork_bytes);
    cudaDeviceSynchronize();
    cudaMalloc((void **) &work_d, lwork_bytes);
    int niters;
    int *info_d;
    cudaMalloc((void**)&info_d, sizeof(int));
    double start = omp_get_wtime();
    status = cusolverDnDDgels(handle, m, n, nrhs, A_T_d, lda, b.getDevicePointer(), ldb, x_d, ldx, work_d, lwork_bytes, &niters, info_d);
    cudaDeviceSynchronize();
    double end = omp_get_wtime();
    //printf("time = %f\n",end-start);
    int info_h = 0;
    cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);

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
    //printf("(n_int, n_ext) = (%d, %d)\n",n_int,n_ext);
    //cudaDeviceSynchronize();
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


    /*cudaMemcpy(x_h, x_d,    A.cols * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < A.cols; i++) {
        printf("x = %e\n",x_h[i]);
    }
    for (int i = 0; i < n_int; i++) {
        C.setHostRealValue(i,x_h[i]);
        C.setHostImagValue(i,x_h[i + n_ext + n_int]);
    }
    
    for (int i = 0; i < n_ext; i++) {
        D.setHostRealValue(i,x_h[i + n_int]);
        D.setHostImagValue(i,x_h[i + n_ext + 2*n_int]);
    }*/
    // Cleanup
    cudaFree(A_T_d);
    cudaFree(x_d);
    //cudaFreeHost(x_h);
    cudaFree(work_d);
    cudaFree(info_d);
    cusolverDnDestroy(handle);
}

void Segment::solveLinearSystem() {

    //C = ComplexMatrix(n_int);
    //D = ComplexMatrix(n_ext);

    char trans;
    int m, n, nrhs, lda, ldb, info, lwork;
    double work_query;
    double *work;

    // Test if it works
    /*double Atest_h[12] = {1, 2, 3,
                        5, 7, 7,
                        9, 10, 11,
                        12, 13, 14}; 
    //double Atest_h[12] = {1, 5, 9, 12, 2, 7, 10, 13, 3, 7, 11, 14}; 

    double btest_h[4];
    btest_h[0] = 14;
    btest_h[1] = 40;
    btest_h[2] = 62;
    btest_h[3] = 80;
    int rows = 4;
    int cols = 3;

    // Define variables for solving linear system
    m = rows;
    n = cols;
    nrhs = 1; 
    lda = m;
    ldb = m;
    int ldx = n;

    // Move to device and transpose
    double * Atest_d, *Atest_T_d, *btest_d, *xtest_h, *xtest_d;
    void * work_d = NULL;

    cudaMalloc((void **) &Atest_d,     rows * cols * sizeof(double));
    cudaMalloc((void **) &Atest_T_d,     rows * cols * sizeof(double));
    cudaMalloc((void **) &btest_d,     rows * sizeof(double));
    cudaMallocHost((void **) &xtest_h,     cols * sizeof(double));
    cudaMalloc((void **) &xtest_d,     cols * sizeof(double));
    cudaMemcpy(Atest_d,    Atest_h,    rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(btest_d,    btest_h,    rows * sizeof(double), cudaMemcpyHostToDevice);

    // Blocks and threads
    dim3 dimBlock(32,32);
    dim3 dimGrid((rows + dimBlock.x - 1)/dimBlock.x, (cols + dimBlock.y - 1)/dimBlock.y);
    transpose<<< dimGrid, dimBlock>>>(Atest_d, Atest_T_d, rows, cols);

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    size_t lwork_bytes = 0;

    cusolverStatus_t status = cusolverDnDDgels_bufferSize(handle, m, n, nrhs, Atest_T_d, lda, btest_d,ldb,xtest_d, ldx, work_d, &lwork_bytes);
    
    cudaMalloc((void **) &work_d, lwork_bytes);
    int niters;
    int *info_d;
    cudaMalloc((void**)&info_d, sizeof(int));

    status = cusolverDnDDgels(handle, m, n, nrhs, Atest_T_d, lda, btest_d,ldb,xtest_d, ldx, work_d, lwork_bytes, &niters, info_d);

    int info_h = 0;
    cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);

    if (info_h == 0) {
        std::cout << "Solution was successful.\n";
    } else {
        std::cout << "Solution failed with info = " << info_h << std::endl;
    }

    cudaMemcpy(xtest_h,    xtest_d,    cols * sizeof(double), cudaMemcpyDeviceToHost);
    for (int j = 0; j < cols; j++) {
        printf("%f\n",xtest_h[j]);
    }
    /*m = rows;
    n = cols;
    nrhs = 1; 
    lda = m;
    ldb = std::max(m, n);
    printf("(cols, rows, ldb) = (%d, %d, %d)\n", cols, rows, ldb);
    lwork = -1;

    // cuBLAS handle creation
    cublasHandle_t handle;
    cublasStatus_t status;
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS initialization failed %d\n",status);
        return;
    }
    double * Atest_d, *Atest_T_d, *btest_d;

    cudaMalloc((void **) &Atest_d,     rows * cols * sizeof(double));
    cudaMalloc((void **) &Atest_T_d,     rows * cols * sizeof(double));
    cudaMalloc((void **) &btest_d,     rows * sizeof(double));
    cudaMemcpy(Atest_d,    Atest_h,    rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(btest_d,    btest_h,    rows * sizeof(double), cudaMemcpyHostToDevice);

    // Blocks and threads
    dim3 dimBlock(32,32);
    dim3 dimGrid((rows + dimBlock.x - 1)/dimBlock.x, (cols + dimBlock.y - 1)/dimBlock.y);
    transpose<<< dimGrid, dimBlock>>>(Atest_d, Atest_T_d, rows, cols);

    double * A_ptr_h = Atest_T_d; // have tried Atest_h
    double * b_ptr_h = btest_d;
    double ** A_ptr_d;
    double ** b_ptr_d;
    cudaMalloc(&A_ptr_d, sizeof(double*));
    cudaMalloc(&b_ptr_d, sizeof(double*));
    cudaMemcpy(A_ptr_d, &A_ptr_h, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(b_ptr_d, &b_ptr_h, sizeof(double*), cudaMemcpyHostToDevice);

    int * info_h;
    int* info_d;
    //cudaMalloc((void**)&info_d, sizeof(int));
    printf("HEJ FRA HER!\n");
    //int *devInfoArray; 
    //cudaMalloc((void**)&devInfoArray,  sizeof(int));
    cudaMallocHost((void **) &info_h,  sizeof(int));
    cudaMalloc((void **) &info_d,     sizeof(int));

    int devInfoArray[1] = { 0 };
        cudaDeviceSynchronize();
    status = cublasDgelsBatched(handle, CUBLAS_OP_N, m, n, nrhs, A_ptr_d, lda, b_ptr_d, ldb, info_h, NULL, 1);
    cudaMemcpy(btest_h,    btest_d,    rows * sizeof(double), cudaMemcpyDeviceToHost);

    //cudaMemcpy(info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("HEJ FRA HER 2!\n");
    cudaDeviceSynchronize();

    printf("b:\n");

    for (int j = 0; j < 4; j++) {
        printf("%f\n",btest_h[j]);
    }

    // Check if cublasDtrmv was successful
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cublasDgelsBatched failed with error code: %d\n", status);
        printf("info = %d\n",*info_h);
    }
    printf("info = %d\n",*info_h);

    // Destroy cuBLAS handle
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS destruction failed\n");
        return;
    }
    

    /*dgels_(&trans, &m, &n, &nrhs, Atest, &lda, btest, &ldb, &work_query, &lwork, &info);
    
    lwork = (int)work_query;
    work = (double*)malloc(lwork * sizeof(double));

    printf("Solving linear system now.\n");
    dgels_(&trans, &m, &n, &nrhs, Atest, &lda, btest, &ldb, work, &lwork, &info);

    printf("b:\n");

    for (int j = 0; j < 4; j++) {
        printf("%f\n",btest[j]);
    }
    double Atest2[12] = {1, 2, 3, 5, 6, 7, 9, 10, 11,12, 13, 14}; 
    printf("Ab:\n");
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            printf("%f ",Atest2[r*cols+c]);
        }
        printf("\n");
    }
    printf("Ab:\n");
    for (int r = 0; r < rows; r++) {
        double val = 0.0;
        for (int c = 0; c < cols; c++) {
            printf("%d ",r*cols+c);
            val += Atest2[r*cols+c]*btest[c];
            printf("%f ",val);
        }
        printf("%f\n",val);
    }
/*printf("b:\n");
    for (int j = 0; j < n_int; j++) {
        printf("%e\n",b.getHostValue(j));
    }*/

    /*trans = 'T';
    m = A.cols;
    n = A.rows;
    nrhs = 1; 
    lda = m;
    ldb = std::max(m, n);
    lwork = -1;*/

    if (true) {

        A.toDevice();
        b.toDevice();

        solveLinearSystem_GPU(A, b, C, D);

        C.toHost();
        D.toHost();
       
        /*// cuBLAS handle creation
        cublasHandle_t handle;
        cublasStatus_t status;
        status = cublasCreate(&handle);

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS initialization failed %d\n",status);
            return;
        }

        A.toDevice();
        b.toDevice();

        m = A.rows;
        n = A.cols;
         nrhs = 1; 
    lda = m;
    ldb = std::max(m, n);
    lwork = -1;
        double * A_T_d;
        cudaMalloc((void **) &A_T_d, A.rows * A.cols * sizeof(double));

        // Blocks and threads
        dim3 dimBlock(32,32);
        dim3 dimGrid((A.rows + dimBlock.x - 1)/dimBlock.x, (A.cols + dimBlock.y - 1)/dimBlock.y);
        transpose<<< dimGrid, dimBlock>>>(A.getDevicePointer(), A_T_d, A.rows, A.cols);

    
        double * A_ptr_h = A_T_d;
        double * b_ptr_h = b.getDevicePointer();
        double ** A_ptr_d;
        double ** b_ptr_d;
        cudaMalloc(&A_ptr_d, sizeof(double*));
        cudaMalloc(&b_ptr_d, sizeof(double*));
        cudaMemcpy(A_ptr_d, &A_ptr_h, sizeof(double*), cudaMemcpyHostToDevice);
        cudaMemcpy(b_ptr_d, &b_ptr_h, sizeof(double*), cudaMemcpyHostToDevice);


        int * info_h;
    int* info_d;
    //cudaMalloc((void**)&info_d, sizeof(int));
  
    //int *devInfoArray; 
    //cudaMalloc((void**)&devInfoArray,  sizeof(int));
    cudaMallocHost((void **) &info_h,  sizeof(int));
    cudaMalloc((void **) &info_d,     sizeof(int));
    double start = omp_get_wtime();
        cudaDeviceSynchronize();
        
        status = cublasDgelsBatched(handle, CUBLAS_OP_N, m, n, nrhs, A_ptr_d, lda, b_ptr_d, ldb, info_h, NULL, 1);
        
        cudaDeviceSynchronize();
        double end = omp_get_wtime();
        printf("time = %f\n",end-start);
        printf("HEJ FRA HER 2!\n");
        // Check if cublasDtrmv was successful
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cublasDgelsBatched failed with error code: %d\n", status);
        }

        // Destroy cuBLAS handle
        status = cublasDestroy(handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("cuBLAS destruction failed\n");
            return;
        }
        b.toHost();*/
    }
    else {

        /*for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                printf("%e ",A.getHostValue(i,j));
            }
            printf("\n");
        }

        for (int j = 0; j < 10; j++) {
            printf("%e\n",b.getHostValue(j));
        }*/
        
        solveLinearSystem_CPU(A, b, C, D);

    }
    


    // Free arrays that we no longer need
    //A.free();
    //b.free();
    //n_x.free();
    //n_y.free();
    //x_test.free();
    //y_test.free();

    /*printf("b:\n");

    for (int j = 0; j < 4; j++) {
        printf("%f\n",btest[j]);
    }*/

    /*printf("C:\n");

    for (int j = 0; j < 10; j++) {
        printf("%e\t + i(%e)\n",C.getHostRealValue(j),C.getHostImagValue(j));
    }/*

    printf("D:\n");

    for (int j = 0; j < n_ext; j++) {
        printf("%e\t + i(%e)\n",D.getHostRealValue(j),D.getHostImagValue(j));
    }*/
    
 
}


}