#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
extern "C" {
#include "../../../lib/Nanostructure.h"
#include "../../../lib/RealMatrix.h"
#include "../../../lib/kernels.h"
using namespace std;


void computeTestPoints(RealMatrix x_test, RealMatrix y_test, Nanostructure nanostructure, int start, int end, double leftStep, double rightStep, int n_top, int n_right, int n_bottom, int n_left, double left_x_value, double right_x_value, bool deviceComputation, bool printOutput) {

    bool host = !deviceComputation;
    bool device = deviceComputation;
    RealMatrix x_test_top       = RealMatrix(n_top,    host, device);
    RealMatrix y_test_top       = RealMatrix(n_top,    host, device);
    RealMatrix x_test_right     = RealMatrix(n_right,  host, device);
    RealMatrix y_test_right     = RealMatrix(n_right,  host, device);
    RealMatrix x_test_bottom    = RealMatrix(n_bottom, host, device);
    RealMatrix y_test_bottom    = RealMatrix(n_bottom, host, device);
    RealMatrix x_test_left      = RealMatrix(n_left,   host, device);
    RealMatrix y_test_left      = RealMatrix(n_left,   host, device);

    // Remove end points
    start     += 1;
    end       -= 1;

    if (deviceComputation) { // GPU
        if (printOutput) printf("Computing test points on the GPU.\n");

        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((n_right + dimBlock.x - 1)/dimBlock.x);

        setConstantKernel<<<dimGrid, dimBlock>>>(x_test_right, 0,  n_right, right_x_value);
        setReversedKernel<<<dimGrid, dimBlock>>>(y_test_right, 0, n_right, rightStep);

        dimGrid.x = (n_left + dimBlock.x - 1)/dimBlock.x;
        setConstantKernel<<<dimGrid, dimBlock>>>(x_test_left, 0, n_left, left_x_value);
        setLinearKernel<<<dimGrid, dimBlock>>>(y_test_left,  0, n_left, leftStep);

        dimGrid.x = (end - start + dimBlock.x - 1)/dimBlock.x;
        setVectorKernel<<<dimGrid, dimBlock>>>(x_test_top, 0,         end - start, nanostructure.x, start);
        setVectorKernel<<<dimGrid, dimBlock>>>(y_test_top, 0,         end - start, nanostructure.f, start);
        setReversedVectorKernel<<<dimGrid, dimBlock>>>(x_test_bottom, 0, end - start, nanostructure.x, start);
        setConstantKernel<<<dimGrid, dimBlock>>>(y_test_bottom, 0,     end - start, 0.0);

        cudaDeviceSynchronize();

        // Combine all points to the same vector
        int shift = 0;
        dimGrid.x = (n_top + dimBlock.x - 1)/dimBlock.x;
        setVectorKernel<<<dimGrid, dimBlock>>>(x_test, shift, n_top,   x_test_top,     0);
        setVectorKernel<<<dimGrid, dimBlock>>>(y_test, shift, n_top,   y_test_top,     0);
        shift += n_top;
        dimGrid.x = (n_right + dimBlock.x - 1)/dimBlock.x;
        setVectorKernel<<<dimGrid, dimBlock>>>(x_test, shift, n_right,  x_test_right,  0);
        setVectorKernel<<<dimGrid, dimBlock>>>(y_test, shift, n_right,  y_test_right,  0);
        shift += n_right;
        dimGrid.x = (n_bottom + dimBlock.x - 1)/dimBlock.x;
        setVectorKernel<<<dimGrid, dimBlock>>>(x_test, shift, n_bottom, x_test_bottom, 0);
        setVectorKernel<<<dimGrid, dimBlock>>>(y_test, shift, n_bottom, y_test_bottom, 0);
        shift += n_bottom;
        dimGrid.x = (n_left + dimBlock.x - 1)/dimBlock.x;
        setVectorKernel<<<dimGrid, dimBlock>>>(x_test, shift, n_left,   x_test_left,   0);
        setVectorKernel<<<dimGrid, dimBlock>>>(y_test, shift, n_left,   y_test_left,   0);

        cudaDeviceSynchronize();

        //x_test.toHost();
        //y_test.toHost();

    }
    else { // CPU
        if (printOutput) printf("Computing test points on the CPU.\n");

        // Compute points along each side
        for (int j = 0; j < n_right; j++) {
            x_test_right.setHostValue(j, right_x_value);
            y_test_right.setHostValue(n_right - j - 1, (j+1)*rightStep);
        }
        
        for (int j = 0; j < n_left; j++) {
            x_test_left.setHostValue(j, left_x_value);
            y_test_left.setHostValue(j, (j+1)*leftStep);
        }

        for (int j = start; j < end; j++) {
            
            x_test_top.setHostValue(j - start, nanostructure.x.getHostValue(j));
            y_test_top.setHostValue(j - start, nanostructure.f.getHostValue(j));
            x_test_bottom.setHostValue(end - j - 1, nanostructure.x.getHostValue(j));
            y_test_bottom.setHostValue(j - start, 0.0);
        }

        // Combine points into combined vector
        int shift = 0;
        for (int j = 0; j < n_top;    j++) {
            x_test.setHostValue(j + shift,x_test_top.getHostValue(j));
            y_test.setHostValue(j + shift,y_test_top.getHostValue(j));
        }
        shift += n_top;
        for (int j = 0; j < n_right;  j++) {
            x_test.setHostValue(j + shift,x_test_right.getHostValue(j));
            y_test.setHostValue(j + shift,y_test_right.getHostValue(j));
        }
        shift += n_right;
        for (int j = 0; j < n_bottom; j++) {
            x_test.setHostValue(j + shift,x_test_bottom.getHostValue(j));
            y_test.setHostValue(j + shift,y_test_bottom.getHostValue(j));
        }
        shift += n_bottom;
        for (int j = 0; j < n_left;   j++) {
            x_test.setHostValue(j + shift,x_test_left.getHostValue(j));
            y_test.setHostValue(j + shift,y_test_left.getHostValue(j));
        }        

    }
    x_test_top.free();      y_test_top.free();
    x_test_right.free();    y_test_right.free();
    x_test_bottom.free();   y_test_bottom.free();
    x_test_left.free();     y_test_left.free();
}

}