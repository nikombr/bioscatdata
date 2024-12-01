#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
extern "C" {
#include "../../../lib/RealMatrix.h"
using namespace std;

__global__ void computeInteriorPointsKernel(RealMatrix x_int, RealMatrix y_int, RealMatrix x_test, RealMatrix y_test, double alpha, int n_top, int n_right, int n_bottom, int n_left, int n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        double xdiff, ydiff, norm;
        int shift;
        
        if (j < n_top - 4) {
            shift = -2;
        }
        else if (j < n_top + n_right - 8) {
            shift = -6;
        }
        else if (j < n_top + n_right + n_bottom - 12) {
            shift = -10;
        }
        else {
            shift = -14;
        }
        xdiff = x_test.getDeviceValue(j - 1 - shift) - x_test.getDeviceValue(j + 1 - shift); // Central difference
        ydiff = y_test.getDeviceValue(j - 1 - shift) - y_test.getDeviceValue(j + 1 - shift); // Central difference
        
        norm = std::sqrt(xdiff*xdiff + ydiff*ydiff);
        xdiff /= norm;
        ydiff /= norm;
        //printf("alpha=%e\n",alpha);
        x_int.setDeviceValue(j, x_test.getDeviceValue(j - shift) - alpha*ydiff);
        y_int.setDeviceValue(j, y_test.getDeviceValue(j - shift) + alpha*xdiff);
    }
}

void computeInteriorPoints(RealMatrix x_int, RealMatrix y_int, RealMatrix x_test, RealMatrix y_test, double alpha, int n_top, int n_right, int n_bottom, int n_left, bool deviceComputation, bool printOutput) {
    int n = x_int.rows;
    if (deviceComputation) { // GPU
        if (printOutput) printf("Computing interior points on the GPU.\n");
        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);
        computeInteriorPointsKernel<<<dimGrid, dimBlock>>>(x_int, y_int, x_test, y_test, alpha, n_top, n_right, n_bottom, n_left, n); 
        cudaDeviceSynchronize();
    }
    else { // CPU
        if (printOutput) printf("Computing interior points on the CPU.\n");
        for (int j = 0; j < n; j++) {
            double xdiff, ydiff, norm;
            int shift;
            
            if (j < n_top - 4) {
                shift = -2;
            }
            else if (j < n_top + n_right - 8) {
                shift = -6;
            }
            else if (j < n_top + n_right + n_bottom - 12) {
                shift = -10;
            }
            else {
                shift = -14;
            }
            xdiff = x_test.getHostValue(j - 1 - shift) - x_test.getHostValue(j + 1 - shift); // Central difference
            ydiff = y_test.getHostValue(j - 1 - shift) - y_test.getHostValue(j + 1 - shift); // Central difference
            
            norm = std::sqrt(xdiff*xdiff + ydiff*ydiff);
            xdiff /= norm;
            ydiff /= norm;
            //printf("alpha=%e\n",alpha);
            x_int.setHostValue(j, x_test.getHostValue(j - shift) - alpha*ydiff);
            y_int.setHostValue(j, y_test.getHostValue(j - shift) + alpha*xdiff);

        }

    }
}


}