#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
extern "C" {
#include "../lib/ComplexMatrix.h"
using namespace std;

ComplexMatrix::ComplexMatrix(int rows, int cols) {

    // Save input parameters
    this->rows       = rows;
    this->cols       = cols;
    this->host       = true;
    this->device     = true;

    // Allocate vectors on host
    cudaMallocHost((void **) &real_h, rows*cols*sizeof(double));
    cudaMallocHost((void **) &imag_h, rows*cols*sizeof(double));

    // Allocate vectors on device
    cudaMalloc((void **) &real_d, rows*cols*sizeof(double));
    cudaMalloc((void **) &imag_d, rows*cols*sizeof(double));
}

ComplexMatrix::ComplexMatrix(int rows, int cols, bool host, bool device) {

    // Save input parameters
    this->rows       = rows;
    this->cols       = cols;
    this->host       = host;
    this->device     = device;

    if (host) {
        // Allocate vectors on host
        cudaMallocHost((void **) &real_h, rows*cols*sizeof(double));
        cudaMallocHost((void **) &imag_h, rows*cols*sizeof(double));
    }

    if (device) {
        // Allocate vectors on device
        cudaMalloc((void **) &real_d, rows*cols*sizeof(double));
        cudaMalloc((void **) &imag_d, rows*cols*sizeof(double));
    }
}

ComplexMatrix::ComplexMatrix(int rows) {

    // Save input parameters
    this->rows       = rows;
    this->cols       = 1;
    this->host       = true;
    this->device     = true;

    // Allocate vectors on host
    cudaMallocHost((void **) &real_h, rows*cols*sizeof(double));
    cudaMallocHost((void **) &imag_h, rows*cols*sizeof(double));

    // Allocate vectors on device
    cudaMalloc((void **) &real_d, rows*cols*sizeof(double));
    cudaMalloc((void **) &imag_d, rows*cols*sizeof(double));
}

ComplexMatrix::ComplexMatrix(int rows, bool host, bool device) {

    // Save input parameters
    this->rows       = rows;
    this->cols       = 1;
    this->host       = host;
    this->device     = device;

    if (host) {
        // Allocate vectors on host
        cudaMallocHost((void **) &real_h, rows*cols*sizeof(double));
        cudaMallocHost((void **) &imag_h, rows*cols*sizeof(double));
    }

    if (device) {
        // Allocate vectors on device
        cudaMalloc((void **) &real_d, rows*cols*sizeof(double));
        cudaMalloc((void **) &imag_d, rows*cols*sizeof(double));
    }
}

void ComplexMatrix::free() {
    cudaError_t err;
    if (host) {
        // Free on host
        err = cudaFreeHost(real_h);

        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on host: " << cudaGetErrorString(err) << std::endl;
        }
        err = cudaFreeHost(imag_h);

        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on host: " << cudaGetErrorString(err) << std::endl;
        }
    }
    if (device) {
        // Free on device
        err = cudaFree(real_d);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on device: " << cudaGetErrorString(err) << std::endl;
        }
        err = cudaFree(imag_d);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on device: " << cudaGetErrorString(err) << std::endl;
        }
    }

}

void ComplexMatrix::toHost() {
    if (host && device) {
        // Send from device to host
        cudaMemcpy(real_h, real_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(imag_h, imag_d, rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    }

}

void ComplexMatrix::toDevice() {

    if (host && device) {
        // Send from host to device
        cudaMemcpy(real_d, real_h, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(imag_d, imag_h, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    }
    
}

void ComplexMatrix::setHostRealValue(int r, double val) {
    real_h[r] = val;
}

void ComplexMatrix::setHostRealValue(int r, int c, double val) {
    real_h[r*cols + c] = val;
}

double ComplexMatrix::getHostRealValue(int r) {
    return real_h[r];
}

double ComplexMatrix::getHostRealValue(int r, int c) {
    return real_h[r*cols + c];
}


void ComplexMatrix::setHostImagValue(int r, double val) {
    imag_h[r] = val;
}

void ComplexMatrix::setHostImagValue(int r, int c, double val) {
    imag_h[r*cols + c] = val;
}


double ComplexMatrix::getHostImagValue(int r) {
    return imag_h[r];
}

double ComplexMatrix::getHostImagValue(int r, int c) {
    return imag_h[r*cols + c];
}


void ComplexMatrix::dumpResult(const char * filename) {
    if (cols == 1) {
        FILE *file;
        file = fopen(filename, "w");
        if (file == NULL) {
            perror("Error opening file");
            return;
        }
        for (int r = 0; r < rows; r++) {
            fprintf(file, "%e\t%e\n", getHostRealValue(r), getHostImagValue(r));
        }
        fclose(file);
    }
    else printf("We do not support saving complex matrices with several columns.\n");
}


}