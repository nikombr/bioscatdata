
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include "../lib/RealMatrix.h"
extern "C" {
using namespace std;

RealMatrix::RealMatrix(int rows) {
     // Save input parameters
    this->rows       = rows;
    this->cols       = 1;
    this->depth      = 1;
    this->host       = true;
    this->device     = true;

    // Allocate vectors on host
    cudaMallocHost((void **) &val_h,    rows*cols*depth*sizeof(double));
    if (val_h == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }

    // Allocate vectors on device
    cudaMalloc((void **) &val_d,    rows*cols*depth*sizeof(double));

    if (val_d == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }
}

RealMatrix::RealMatrix(int rows, bool host, bool device) {
     // Save input parameters
    this->rows       = rows;
    this->cols       = 1;
    this->depth      = 1;
    this->host       = host;
    this->device     = device;

    // Allocate vectors on host
    if (host) {
        cudaMallocHost((void **) &val_h,    rows*cols*depth*sizeof(double));
        if (val_h == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            return;
        }
    }

    // Allocate vectors on device
    if (device) {
        cudaMalloc((void **) &val_d,    rows*cols*depth*sizeof(double));

        if (val_d == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            return;
        }
    }
}

RealMatrix::RealMatrix(int rows, int cols) {

    // Save input parameters
    this->rows       = rows;
    this->cols       = cols;
    this->depth      = 1;
    this->host       = true;
    this->device     = true;

    // Allocate vectors on host
    cudaMallocHost((void **) &val_h,    rows*cols*depth*sizeof(double));
    if (val_h == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }

    // Allocate vectors on device
    cudaMalloc((void **) &val_d,    rows*cols*depth*sizeof(double));

    if (val_d == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }
}

RealMatrix::RealMatrix(int rows, int cols, bool host, bool device) {

    // Save input parameters
    this->rows       = rows;
    this->cols       = cols;
    this->depth      = 1;
    this->host       = host;
    this->device     = device;

    if (host) {
        // Allocate vectors on host
        cudaMallocHost((void **) &val_h,    rows*cols*depth*sizeof(double));
        if (val_h == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            return;
        }
    }

    if (device) {
        // Allocate vectors on device
        cudaMalloc((void **) &val_d,    rows*cols*depth*sizeof(double));

        if (val_d == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            return;
        }
    }
}

RealMatrix::RealMatrix(int rows, int cols, int depth) {

    // Save input parameters
    this->rows       = rows;
    this->cols       = cols;
    this->depth      = depth;
    this->host       = true;
    this->device     = true;

    // Allocate vectors on host
    cudaMallocHost((void **) &val_h,    rows*cols*depth*sizeof(double));
    if (val_h == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }

    // Allocate vectors on device
    cudaMalloc((void **) &val_d,    rows*cols*depth*sizeof(double));

    if (val_d == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return;
    }
}

RealMatrix::RealMatrix(int rows, int cols, int depth, bool host, bool device) {

    // Save input parameters
    this->rows       = rows;
    this->cols       = cols;
    this->depth      = depth;
    this->host       = host;
    this->device     = device;

    if (host) {
        // Allocate vectors on host
        cudaMallocHost((void **) &val_h,    rows*cols*depth*sizeof(double));
        if (val_h == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            return;
        }
    }

    if (device) {
        // Allocate vectors on device
        cudaMalloc((void **) &val_d,    rows*cols*depth*sizeof(double));

        if (val_d == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            return;
        }
    }
}

void RealMatrix::free() {
    
    cudaError_t err;
    // Free on host
    if (host) {
        err = cudaFreeHost(val_h);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on host: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Free on device
    if (device) {
        err = cudaFree(val_d);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on device: " << cudaGetErrorString(err) << std::endl;
        }
    }

    //printf("Destructed real matrices!\n");

}

void RealMatrix::toHost() {

    if (host && device) {
        // Send from device to host
        cudaMemcpy(val_h,    val_d,    depth * rows * cols * sizeof(double), cudaMemcpyDeviceToHost);
    }
    else {
        printf("You cannot send between the host and device when both are not allocated.\n");
    }

}

void RealMatrix::toDevice() {

    if (host && device) {
        // Send from host to device
        cudaMemcpy(val_d,    val_h,    depth * rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    }
    else {
        printf("You cannot send between the host and device when both are not allocated.\n");
    }
    
}

void RealMatrix::setHostValue(int r, double val) {
    val_h[r] = val;
}

void RealMatrix::setHostValue(int r, int c, double val) {
    val_h[r*cols + c] = val;
}

void RealMatrix::setHostValue(int r, int c, int d, double val) {
    val_h[r * (cols * depth) + c * depth + d] = val;
}

double RealMatrix::getHostValue(int r) {
    return val_h[r];
}

double RealMatrix::getHostValue(int r, int c) {
    return val_h[r*cols + c];
}

double RealMatrix::getHostValue(int r, int c, int d) {
    return val_h[r * (cols * depth) + c * depth + d];
}


void RealMatrix::dumpResult(const char * filename) {
    if (cols == 1) {
        FILE *file;
        file = fopen(filename, "w");
        if (file == NULL) {
            perror("Error opening file");
            return;
        }
        for (int r = 0; r < rows; r++) {
            fprintf(file, "%e\n", getHostValue(r));
        }
        fclose(file);
    }
    else printf("We do not support saving real matrices with several columns.\n");
}



}