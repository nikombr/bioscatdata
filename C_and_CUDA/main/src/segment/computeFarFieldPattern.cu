#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include "../../lib/Segment.h"
#include "../../lib/RealMatrix.h"
extern "C" {
using namespace std;

#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

__global__ void computeFarFieldPatternKernel(ComplexMatrix F, ComplexMatrix C, RealMatrix phi, RealMatrix x_int, RealMatrix y_int, int rows, int cols, double k) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r < rows) {
        double abs_int, xdiff, ydiff, exp_real, exp_imag, rho_int, phi_int, x, y, C_real, C_imag, phiIdx, val_real,val_imag;
        val_real = 0;
        val_imag = 0;
        phiIdx = phi.getDeviceValue(r);
        for (int c = 0; c < cols; c++) {

            // Get data
            x = x_int.getDeviceValue(c);
            y = y_int.getDeviceValue(c);
            
            phi_int = atan2(y, x);
            rho_int = sqrt(x * x + y * y);
            C_real = C.getDeviceRealValue(c);
            C_imag = C.getDeviceImagValue(c);

            // Compute first Hankel functions
            exp_real = cos(k * rho_int * cos(phiIdx - phi_int));
            exp_imag = sin(k * rho_int * cos(phiIdx - phi_int));

            // Compute the first field
            val_real += C_real * exp_real - C_imag * exp_imag;
            val_imag += C_real * exp_imag + C_real * exp_imag;
        }
        F.setDeviceRealValue(r, val_real);
        F.setDeviceImagValue(r, val_imag);
    }
}

void Segment::computeFarFieldPattern(RealMatrix phi) {

    int rows = phi.rows;
    int cols = y_int.rows;
    double k0 = constants.k0;

    if (deviceComputation) {

        // Blocks and threads
        dim3 dimBlock(32);
        dim3 dimGrid((rows + dimBlock.x - 1)/dimBlock.x);
        
        computeFarFieldPatternKernel<<<dimGrid, dimBlock>>>(F, C, phi, x_int, y_int, rows, cols, k0);
        cudaCheckError();
        cudaDeviceSynchronize();
        
    }
    else {
        // not implemented
    }

}

}