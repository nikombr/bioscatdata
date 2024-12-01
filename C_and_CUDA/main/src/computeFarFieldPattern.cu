#include <stdlib.h>
#include <stdio.h>
//#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <omp.h>
//#include <cblas.h>
#include <math.h>
#include "../lib/RealMatrix.h"
#include "../lib/BioScat.h"
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
extern "C" {

__global__ void moveFieldsFromSegments(ComplexMatrix field, ComplexMatrix subField, int rows) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < rows) {
        field.setDeviceRealValue(k, field.getDeviceRealValue(k) + subField.getDeviceRealValue(k));
        field.setDeviceImagValue(k, field.getDeviceImagValue(k) + subField.getDeviceImagValue(k));
    }
}

__global__ void combineFields(ComplexMatrix pol1, ComplexMatrix pol2, RealMatrix field, int rows, double beta) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < rows) {
        double Efar1sq = pol1.getDeviceRealValue(k) * pol1.getDeviceRealValue(k) + pol1.getDeviceImagValue(k) * pol1.getDeviceImagValue(k);
        double Efar2sq = pol2.getDeviceRealValue(k) * pol2.getDeviceRealValue(k) + pol2.getDeviceImagValue(k) * pol2.getDeviceImagValue(k);
        double cosBeta = cos(beta);
        double sinBeta = sin(beta);
        double val = cosBeta * cosBeta * Efar1sq + sinBeta * sinBeta * Efar1sq;
        field.setDeviceValue(k, sqrt(val));
    }
}

void BioScat::combineFarFieldPattern() {

    int n = phi_obs.rows;

    if (deviceComputation) {
     
        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);

        combineFields<<<dimGrid, dimBlock>>>(F_pol[0], F_pol[1], F, n, beta);
        cudaDeviceSynchronize();
        

    }
    else {
        // not implemented
    }

}


void BioScat::computeFarFieldPattern() {

    int n = phi_obs.rows;

    for (int i = 0; i < num_segments; i++) {
        
        double start_inner = omp_get_wtime();
        
        segments[i].computeFarFieldPattern(phi_obs);

        double end_inner = omp_get_wtime();

        if (printOutput) printf("\nIt took %.4e seconds to compute the scattered far field field pattern for segment %d.\n\n",end_inner - start_inner, i + 1);
        
    }

    if (deviceComputation) {
     
        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);

        if (polarisation == 1) {
            
            for (int i = 0; i < num_segments; i++) {
                moveFieldsFromSegments<<<dimGrid, dimBlock>>>(F_pol[0], segments[i].F, n);
                cudaDeviceSynchronize();
            }
        }
        else if (polarisation == 2) {
            for (int i = 0; i < num_segments; i++) {
                moveFieldsFromSegments<<<dimGrid, dimBlock>>>(F_pol[1], segments[i].F, n);
                cudaDeviceSynchronize();
            }
        }
        

    }
    else {
        // not implemented
    }

}


}