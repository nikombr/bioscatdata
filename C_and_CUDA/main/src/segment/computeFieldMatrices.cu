#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include "../../lib/Segment.h"
extern "C" {
#include "../../lib/RealMatrix.h"
using namespace std;

#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

__host__ __device__ double H02_real(double x) {
    // Computes real part of Hankel function of order zero and second kind
    int n = 0;
    double Jn = jn(n, x); // Compute Bessel functions of the first (Jn) 
    return Jn;
}

__host__ __device__ double H02_imag(double x) {
    // Computes imaginary part of Hankel function of order zero and second kind
    int n = 0;
    double Yn = yn(n, x); // Compute Bessel functions of the second (Yn) kind
    return -Yn;
}


__host__ __device__ double H12_real(double x) {
    // Computes real part of Hankel function of order one and second kind
    int n = 1;
    double Jn = jn(n, x); // Compute Bessel functions of the first (Jn) 
    return Jn; 
}


 __host__ __device__ double H12_imag(double x) {
    // Computes imaginary part of Hankel function of order one and second kind
    int n = 1;
    double Yn = yn(n, x); // Compute Bessel functions of the second (Yn) kind
    return -Yn;
}

__global__ void computeFieldMatricesKernel(ComplexMatrix F1, ComplexMatrix F2, ComplexMatrix F3, RealMatrix x, RealMatrix x_aux, RealMatrix y, RealMatrix y_aux, double const1, double const2, int rows, int cols, double k) {
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    if (r < rows && c < cols) {
        double abs_aux, xdiff, ydiff, H_real, H_imag;

        // Get data
        xdiff   = x.getDeviceValue(r) - x_aux.getDeviceValue(c);
        ydiff   = y.getDeviceValue(r) - y_aux.getDeviceValue(c);
        abs_aux = std::sqrt(xdiff * xdiff + ydiff * ydiff);

        // Compute first Hankel functions
        H_real = H02_real(k * abs_aux);
        H_imag = H02_imag(k * abs_aux);

        // Compute the first field
        F1.setDeviceRealValue(r, c, const1 * H_real);
        F1.setDeviceImagValue(r, c, const1 * H_imag);

        // Compute second Hankel functions
        H_real = H12_real(k * abs_aux);
        H_imag = H12_imag(k * abs_aux);

        // Compute the second field
        F2.setDeviceRealValue(r, c,   const2 * 1/abs_aux * H_imag * ydiff);
        F2.setDeviceImagValue(r, c, - const2 * 1/abs_aux * H_real * ydiff);

        // Compute the third field
        F3.setDeviceRealValue(r, c, - const2 * 1/abs_aux * H_imag * xdiff);
        F3.setDeviceImagValue(r, c,   const2 * 1/abs_aux * H_real * xdiff);
    }
}



void Segment::computeScatteredFieldMatrices(RealMatrix x, RealMatrix y, bool far_field_approximation) {
    
    int rows = y.rows;
    int cols = y_int.rows;

    ComplexMatrix F1, F2, F3;
    double const1, const2, k0;

    k0 = constants.k0;

    if (polarisation == 1) {
        F1 = E_scat_matrix.z;
        F2 = H_scat_matrix.x;
        F3 = H_scat_matrix.y;
        const1 = 1.0;
        const2 = 1/constants.eta0;
    }
    else if (polarisation == 2) {
        F1 = H_scat_matrix.z;
        F2 = E_scat_matrix.x;
        F3 = E_scat_matrix.y;
        const1 = - 1.0;
        const2 = 1/constants.eta0;
    } 
    else {
        printf("Please input 1 or 2 for the polarisation in \"computeScatteredFieldMatrices\"!\n");
        return;
    }
    if (deviceComputation) {

        // Blocks and threads
        dim3 dimBlock(32,16);
        dim3 dimGrid((rows + dimBlock.x - 1)/dimBlock.x, (cols + dimBlock.y - 1)/dimBlock.y);
        
        computeFieldMatricesKernel<<<dimGrid, dimBlock>>>(F1, F2, F3, x, x_int, y, y_int, const1, const2, rows, cols, k0);
        cudaCheckError();
        cudaDeviceSynchronize();
        
    }
    else {
        /*#pragma omp parallel for collapse(2) 
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double abs_int, abs_int_ref, xdiff, ydiff, ydiff_ref, H_real, H_imag, H_real_ref, H_imag_ref, val;

                // Get data
                xdiff       = x.getHostValue(r) - x_int.getHostValue(c);
                ydiff       = y.getHostValue(r) - y_int.getHostValue(c);
                ydiff_ref   = y.getHostValue(r) + y_int.getHostValue(c);
                abs_int     = std::sqrt(xdiff * xdiff + ydiff     * ydiff);
                abs_int_ref = std::sqrt(xdiff * xdiff + ydiff_ref * ydiff_ref);

                // Compute first Hankel functions
                H_real     = H02_real(k0 * abs_int);
                H_real_ref = H02_real(k0 * abs_int_ref);
                H_imag     = H02_imag(k0 * abs_int);
                H_imag_ref = H02_imag(k0 * abs_int_ref);
                
                val = H_real + Gamma_ref * H_real_ref;
                F1.setHostRealValue(r, c, val);
                val = H_imag + Gamma_ref * H_imag_ref;
                F1.setHostImagValue(r, c, val);

                // Compute second Hankel functions
                H_real     = H12_real(k0 * abs_int);
                H_real_ref = H12_real(k0 * abs_int_ref);
                H_imag     = H12_imag(k0 * abs_int);
                H_imag_ref = H12_imag(k0 * abs_int_ref);

                val = constant * (1/abs_int      * H_imag     * ydiff + \
                        Gamma_ref * 1/abs_int_ref * H_imag_ref * ydiff_ref);
                F2.setHostRealValue(r, c, val);
                val = -constant * (1/abs_int     * H_real     * ydiff + \
                        Gamma_ref * 1/abs_int_ref * H_real_ref * ydiff_ref);
                F2.setHostImagValue(r, c, val);

                val = -constant * xdiff * (1/abs_int      * H_imag      + \
                                Gamma_ref * 1/abs_int_ref  * H_imag_ref);
                F3.setHostRealValue(r, c, val);
                val = constant * xdiff * (1/abs_int     * H_real      + \
                            Gamma_ref * 1/abs_int_ref * H_real_ref);
                F3.setHostImagValue(r, c, val);
            }
        }*/
    }
}

void Segment::computeInteriorFieldMatrices(RealMatrix x, RealMatrix y) {
    
    int rows = y.rows;
    int cols = y_ext.rows;

    ComplexMatrix F1, F2, F3;
    double const1, const2, k1;

    k1 = constants.k1;

    if (polarisation == 1) {
        F1 = E_int_matrix.z;
        F2 = H_int_matrix.x;
        F3 = H_int_matrix.y;
        const1 = 1.0;
        const2 = constants.n1/constants.eta0;
    }
    else if (polarisation == 2) {
        F1 = H_int_matrix.z;
        F2 = E_int_matrix.x;
        F3 = E_int_matrix.y;
        const1 = - 1.0;
        const2 = 1.0/(constants.eta0*constants.n1);
    } 
    else {
        printf("Please input 1 or 2 for the polarisation!\n");
    }

    if (deviceComputation) {
        // Blocks and threads
        dim3 dimBlock(32,16);
        dim3 dimGrid((rows + dimBlock.x - 1)/dimBlock.x, (cols + dimBlock.y - 1)/dimBlock.y);
        
        computeFieldMatricesKernel<<<dimGrid, dimBlock>>>(F1, F2, F3, x, x_ext, y, y_ext, const1, const2, rows, cols, k1);
        cudaCheckError();
        cudaDeviceSynchronize();

    }
    else {
        /*#pragma omp parallel for collapse(2) 
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                double abs_ext, xdiff, ydiff, H_real, H_imag, val;

                // Get data
                xdiff   = x.getHostValue(r) - x_ext.getHostValue(c);
                ydiff   = y.getHostValue(r) - y_ext.getHostValue(c);
                abs_ext = std::sqrt(xdiff*xdiff + ydiff*ydiff);

                // Compute first Hankel functions
                H_real = H02_real(k1 * abs_ext);
                H_imag = H02_imag(k1 * abs_ext);
                
                val = H_real;
                F1.setHostRealValue(r, c, val);
                val = H_imag;
                F1.setHostImagValue(r, c, val);

                // Compute second Hankel functions
                H_real = H12_real(k1 * abs_ext);
                H_imag = H12_imag(k1 * abs_ext);

                val =   constant * 1/abs_ext * ydiff * H_imag;
                F2.setHostRealValue(r, c, val);
                val = - constant * 1/abs_ext * ydiff * H_real;
                F2.setHostImagValue(r, c, val);

                val = -constant * 1/abs_ext * xdiff * H_imag;
                F3.setHostRealValue(r, c, val);
                val =  constant * 1/abs_ext * xdiff * H_real;
                F3.setHostImagValue(r, c, val);
            }
        }*/
    }
}


}