#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
extern "C" {
#include "../../../lib/Nanostructure.h"
#include "../../../lib/RealMatrix.h"
#include "../../../lib/kernels.h"
using namespace std;

__global__ void computeExteriorPointsAndNormalVectorsKernel(RealMatrix x_ext, RealMatrix y_ext, RealMatrix n_x, RealMatrix n_y, RealMatrix x_temp, RealMatrix y_temp, int n, double alpha, int n_top) {
    int j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    if (j < n + 1) {
        double xdiff, ydiff, norm;

        xdiff = x_temp.getDeviceValue(j - 1) - x_temp.getDeviceValue(j + 1); // Central difference
        ydiff = y_temp.getDeviceValue(j - 1) - y_temp.getDeviceValue(j + 1); // Central difference
        
        norm = std::sqrt(xdiff*xdiff + ydiff*ydiff);
        xdiff /= norm;
        ydiff /= norm;

        x_ext.setDeviceValue(j - 1, x_temp.getDeviceValue(j) + alpha*ydiff);
        y_ext.setDeviceValue(j - 1, y_temp.getDeviceValue(j) - alpha*xdiff);

        // Top: Set normal vectors
        if (j >= 2 && j < n_top + 2) {
            n_x.setDeviceValue(j - 2,   ydiff);
            n_y.setDeviceValue(j - 2, - xdiff);
        }
    }
}

void computeExteriorPointsAndNormalVectors(RealMatrix x_ext, RealMatrix y_ext, RealMatrix n_x, RealMatrix n_y, Nanostructure nanostructure, int start, int end, double alpha, double leftStep, double rightStep, int leftNum, int rightNum, int n_top, int n_right, int n_bottom, int n_left, double left_x_value, double right_x_value, bool deviceComputation, bool printOutput) {

    // Allocate array for temporary points to compute exterior auxiliary points
    bool host = !deviceComputation;
    bool device = deviceComputation;
    RealMatrix x_temp = RealMatrix(x_ext.rows + 2, host, device);
    RealMatrix y_temp = RealMatrix(y_ext.rows + 2, host, device);

    if (deviceComputation) { // GPU
        if (printOutput) printf("Computing exterior test points on the GPU.\n");

        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((rightNum + dimBlock.x - 1)/dimBlock.x);

        setConstantKernel<<<dimGrid, dimBlock>>>(x_temp, end - start, rightNum, right_x_value);
        setReversedKernel<<<dimGrid, dimBlock>>>(y_temp, end - start, rightNum, rightStep);

        dimGrid.x = (leftNum + dimBlock.x - 1)/dimBlock.x;
        setConstantKernel<<<dimGrid, dimBlock>>>(x_temp, 2*(end - start - 1) + rightNum + 1, leftNum, left_x_value);
        setLinearKernel2<<<dimGrid, dimBlock>>>(y_temp,  2*(end - start - 1) + rightNum + 1, leftNum, leftStep);

        setConstantKernel<<<1, 1>>>(x_temp, 0, 1, left_x_value);
        setConstantKernel<<<1, 1>>>(y_temp, 0, 1, (leftNum - 1)*leftStep);

        int n = end - 1 - start;
        dimGrid.x = (n + dimBlock.x - 1)/dimBlock.x;
        setVectorKernel<<<dimGrid, dimBlock>>>(        x_temp, 1,                    n, nanostructure.x, start);
        setReversedVectorKernel<<<dimGrid, dimBlock>>>(x_temp, end - start + rightNum, n, nanostructure.x, start+1);
        setVectorKernel<<<dimGrid, dimBlock>>>(        y_temp, 1,                    n, nanostructure.f, start);
        setConstantKernel<<<dimGrid, dimBlock>>>(      y_temp, end - start + rightNum, n, 0.0);

        cudaDeviceSynchronize();

        dimGrid.x = (x_ext.rows + dimBlock.x - 1)/dimBlock.x;
        computeExteriorPointsAndNormalVectorsKernel<<<dimGrid, dimBlock>>>(x_ext, y_ext, n_x, n_y, x_temp, y_temp, x_ext.rows, alpha, n_top);

        // Set normal vectors
        dimGrid.x = (n_right + dimBlock.x - 1)/dimBlock.x;
        setConstantKernel<<<dimGrid, dimBlock>>>(n_x, n_top, n_right, 1.0);
        setConstantKernel<<<dimGrid, dimBlock>>>(n_y, n_top, n_right, 0.0);

        dimGrid.x = (n_bottom + dimBlock.x - 1)/dimBlock.x;
        setConstantKernel<<<dimGrid, dimBlock>>>(n_x, n_top + n_right, n_bottom, 0.0);
        setConstantKernel<<<dimGrid, dimBlock>>>(n_y, n_top + n_right, n_bottom, 1.0);

        dimGrid.x = (n_left + dimBlock.x - 1)/dimBlock.x;
        setConstantKernel<<<dimGrid, dimBlock>>>(n_x, n_top + n_right + n_bottom, n_left, 1.0);
        setConstantKernel<<<dimGrid, dimBlock>>>(n_y, n_top + n_right + n_bottom, n_left, 0.0);

        cudaDeviceSynchronize();
    }
    else { // CPU
        if (printOutput) printf("Computing exterior test points on the CPU.\n");

        for (int j = 0; j < rightNum; j++) {
            x_temp.setHostValue(end - start + j, right_x_value);
            y_temp.setHostValue(end - start + rightNum - j - 1, (j+1)*rightStep);
        }
        
        for (int j = 0; j < leftNum + 1; j++) {
            x_temp.setHostValue(2*(end - start - 1) + rightNum + j + 1, left_x_value);
            y_temp.setHostValue(2*(end - start - 1) + rightNum + j + 1, j*leftStep);
        }

        x_temp.setHostValue(0, left_x_value);
        y_temp.setHostValue(0, (leftNum - 1)*leftStep);

        for (int j = start; j < end - 1; j++) {
            
            x_temp.setHostValue(j - start + 1, nanostructure.x.getHostValue(j));
            x_temp.setHostValue(end - start - 1 + rightNum + end - j - 1, nanostructure.x.getHostValue(j+1));
            y_temp.setHostValue(j - start + 1, nanostructure.f.getHostValue(j));
            y_temp.setHostValue(end - start + rightNum + j - start, 0.0);
        }

        for (int j = 1; j < x_ext.rows + 1; j++) {
            double xdiff, ydiff, norm;

            xdiff = x_temp.getHostValue(j - 1) - x_temp.getHostValue(j + 1); // Central difference
            ydiff = y_temp.getHostValue(j - 1) - y_temp.getHostValue(j + 1); // Central difference
            
            norm = std::sqrt(xdiff*xdiff + ydiff*ydiff);
            xdiff /= norm;
            ydiff /= norm;

            x_ext.setHostValue(j - 1, x_temp.getHostValue(j) + alpha*ydiff);
            y_ext.setHostValue(j - 1, y_temp.getHostValue(j) - alpha*xdiff);

            // Top: Set normal vectors
            if (j >= 2 && j < n_top + 2) {
                n_x.setHostValue(j - 2,   ydiff);
                n_y.setHostValue(j - 2, - xdiff);
            }
        }
        
        // Right side: Set normal vectors
        for (int j = n_top; j < n_top + n_right; j++) {
            
            n_x.setHostValue(j, 1.0);
            n_y.setHostValue(j, 0.0);
        }

        // Bottom: Set normal vectors
        for (int j = n_top + n_right; j < n_top + n_right + n_bottom; j++) {
            n_x.setHostValue(j, 0.0);
            n_y.setHostValue(j, 1.0);
        }

        // Left side: Set normal vectors
        for (int j = n_top + n_right + n_bottom; j < n_top + n_right + n_bottom + n_left; j++) {
            n_x.setHostValue(j, 1.0);
            n_y.setHostValue(j, 0.0);
        }
    }

    x_temp.free();
    y_temp.free();
}


}