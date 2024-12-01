#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
extern "C" {
#include "../../../lib/RealMatrix.h"
using namespace std;

__global__ void setConstantKernel(RealMatrix output, int output_shift, int n, double value) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        output.setDeviceValue(j + output_shift, value);
    }
}

__global__ void setLinearKernel(RealMatrix output, int output_shift, int n, double value) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        output.setDeviceValue(j + output_shift, (j + 1) * value);
    }
}

__global__ void setLinearKernel2(RealMatrix output, int output_shift, int n, double value) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        output.setDeviceValue(j + output_shift, j * value);
    }
}

__global__ void setReversedKernel(RealMatrix output, int output_shift, int n, double value) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        output.setDeviceValue(output_shift + n - j - 1, (j + 1) * value);
    }
}

__global__ void setReversedVectorKernel(RealMatrix output, int output_shift, int n, RealMatrix input, int input_shift) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        output.setDeviceValue(output_shift + n - j - 1, input.getDeviceValue(j + input_shift));
    }
}

__global__ void setVectorKernel(RealMatrix output, int output_shift, int n, RealMatrix input, int input_shift) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        output.setDeviceValue(j + output_shift, input.getDeviceValue(j + input_shift));
        
    }
}


}