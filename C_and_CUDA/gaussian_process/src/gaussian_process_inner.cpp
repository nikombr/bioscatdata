#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
extern "C" {
#include "../lib/GaussianProcess.h"
#include <cuda_runtime_api.h>
#include <ctime>

void print_matrix(double **M_h, double * M_log, bool device, int n) {
    if (n < 21) {
        if (device) {
            cudaMemcpy(*M_h, M_log, n * n * sizeof(double), cudaMemcpyDeviceToHost);
        }
        printf("\n");
        printf("<");
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                if (i != n-1) printf("%.8f, ",M_h[k][i]);
                else printf("%.8f",M_h[k][i]);
            }
            if (k != n-1) printf(";\n");
            
        }
        printf(">");
        printf("\n");
        printf("\n");
    }

}


void gaussian_process_inner(double * x, double * y, int n, double * hyper, int num, int dim, int dev, int type_covfunc) {
    double start, stop;


    start = omp_get_wtime();
    GaussianProcess GP = GaussianProcess(x, y, n, hyper, num, dim, dev, type_covfunc);
    stop = omp_get_wtime();

    printf("Initialization and allocation: %.4f seconds\n\n", stop - start);

    start = omp_get_wtime();
    GP.covariance_matrix();
    stop = omp_get_wtime();

    printf("Computing covariance matrix: %.4f seconds\n\n", stop - start);
    
    print_matrix(GP.M_h, GP.M_log, GP.device, n);

    start = omp_get_wtime();
    GP.cholesky();
    stop = omp_get_wtime();

    printf("Cholesky factorization: %.4f seconds\n\n", stop - start);

    print_matrix(GP.M_h, GP.M_log, GP.device,n);

    start = omp_get_wtime();
    FILE *file;
    file = fopen("../../Data/gaussian_process_realisations/output.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    // Seed the random number generator with the current time
    srand(time(NULL));
    //srand(0);
    for (int k = 0; k < 20000; k++) {
        GP.realisation();
        for (int j = 0; j < n; j++) {
            fprintf(file, "%.4f ", GP.p_h[j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
    stop = omp_get_wtime();

    printf("Computation of random vector and realisation: %.4f seconds\n\n", stop - start);



}

}

