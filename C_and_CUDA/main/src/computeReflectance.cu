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
using namespace std;

__host__ __device__ double getSquaredNorm(double x_real, double x_imag, double y_real, double y_imag, double z_real, double z_imag) {
    double x2 = x_real*x_real + x_imag*x_imag;
    double y2 = y_real*y_real + y_imag*y_imag;
    double z2 = z_real*z_real + z_imag*z_imag;
    return x2 + y2 + z2;
}

__global__ void computeReflectanceKernel(RealMatrix reflectance, Field E_inc, Field E_ref, Field E_scat, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        double numerator   = getSquaredNorm(E_inc.x.getDeviceRealValue(i),                                  E_inc.x.getDeviceImagValue(i),
                                            E_inc.y.getDeviceRealValue(i),                                  E_inc.y.getDeviceImagValue(i),
                                            E_inc.z.getDeviceRealValue(i),                                  E_inc.z.getDeviceImagValue(i));
        double denominator = getSquaredNorm(E_ref.x.getDeviceRealValue(i) + E_scat.x.getDeviceRealValue(i), E_ref.x.getDeviceImagValue(i) + E_scat.x.getDeviceImagValue(i),
                                            E_ref.y.getDeviceRealValue(i) + E_scat.y.getDeviceRealValue(i), E_ref.y.getDeviceImagValue(i) + E_scat.y.getDeviceImagValue(i),
                                            E_ref.z.getDeviceRealValue(i) + E_scat.z.getDeviceRealValue(i), E_ref.z.getDeviceImagValue(i) + E_scat.z.getDeviceImagValue(i));
        reflectance.setDeviceValue(i, numerator/denominator);
    }

}


void BioScat::computeReflectance() {

    if (deviceComputation) {
        int n = x_obs.rows;
        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);
        computeReflectanceKernel<<<dimGrid, dimBlock>>>(reflectance, E_inc, E_ref, E_scat, n);
        cudaDeviceSynchronize();
        //reflectance.toHost();
    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < x_obs.rows; i++) {
            double numerator = getSquaredNorm(E_inc.x.getHostRealValue(i), E_inc.x.getHostImagValue(i),
                                            E_inc.y.getHostRealValue(i), E_inc.y.getHostImagValue(i),
                                            E_inc.z.getHostRealValue(i), E_inc.z.getHostImagValue(i));
            double denominator = getSquaredNorm(E_ref.x.getHostRealValue(i) + E_scat.x.getHostRealValue(i), E_ref.x.getHostImagValue(i) + E_scat.x.getHostImagValue(i),
                                                E_ref.y.getHostRealValue(i) + E_scat.y.getHostRealValue(i), E_ref.y.getHostImagValue(i) + E_scat.y.getHostImagValue(i),
                                                E_ref.z.getHostRealValue(i) + E_scat.z.getHostRealValue(i), E_ref.z.getHostImagValue(i) + E_scat.z.getHostImagValue(i));
            reflectance.setHostValue(i, numerator/denominator);
        }
    }
}



}