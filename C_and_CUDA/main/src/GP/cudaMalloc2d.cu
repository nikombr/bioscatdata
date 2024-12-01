// This file is reused from the HPC course and the advanced special course in HPC

#include <stdlib.h>
extern "C" {
void
host_malloc_2d(double ***B, int n) {

    double **p;
    double *a;

    if (n <= 0) {
        B = NULL;
        return;
    }

    cudaMallocHost(&p, n * sizeof(double *));
    //double **A = malloc(n * sizeof(double *));
    
    if (p == NULL) {
        B = NULL;
        return;
    }

    cudaMallocHost(&a, n * n * sizeof(double));
    //A[0] = malloc(n*n*sizeof(double));

    if (a == NULL) {
        cudaFreeHost(p);
        B = NULL;
        return;
    }

    for (int i = 0; i < n; i++) p[i] = a + i * n;

    *B = p;
}

void
host_free_2d(double **B) {
    cudaFreeHost(B[0]);
    cudaFreeHost(B);
}

__global__ void mallocLoops(double**p, double *a, int n) {

    for (int i = 0; i < n; i++) p[i] = a + i * n;

}

void
device_malloc_2d(double ***B,double **b,int n) {

    double **p;
    double *a;

    if (n <= 0) {
        B = NULL;
        return;
    }

    cudaMalloc(&p, n * sizeof(double *));
    
    if (p == NULL) {
        B = NULL;
        return;
    }

    cudaMalloc(&a, n * n * sizeof(double));


    if (a == NULL) {
        cudaFree(p);
        B = NULL;
        return;
    }

    mallocLoops<<<1,1>>>(p, a, n);
    cudaDeviceSynchronize();

    *B = p;
    *b = a;
}

void device_free_2d(double **B,double *b) {
    cudaFree(b);
    cudaFree(B);
}
}