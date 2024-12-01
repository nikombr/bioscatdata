
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
extern "C" {
#include "../../lib/GP/GaussianProcess.h"
#include "../../lib/GP/cudaMalloc2d.h"
using namespace std;

GaussianProcess::GaussianProcess() {

}

GaussianProcess::GaussianProcess(int n, double* hyper, int num, int type_covfunc) {
    int dim = 1;
    int dev = 1;
    double * y = NULL; // 2D problem
    double * x; 
    cudaMallocHost((void **) &x, n*sizeof(double));
    double step = 10 / ((double) n - 1);
    for (int i = 0; i < n; i++) {
        x[i] = i * step;
    }

    if (dim == 1) {
        y = NULL;
    }

    // Save input parameters
    this->x_h       = x;
    this->y_h       = y;
    this->n         = n;
    this->hyper_h   = hyper;
    this->num       = num;
    this->dim       = dim;
    this->type_covfunc = type_covfunc;

    // Check if device is available
    int temp;
    cudaError_t cudaSuccess =  cudaGetDeviceCount(&temp);
    //device = (!cudaSuccess && temp > 0) ? true : false;
    //printf("devices = %d\n",cudaSuccess);

    device = (!cudaSuccess && temp > 0 && dev == 1) ? true : false;
    string location = device ? "device" : "host";

    printf("--------------------------------------\n");
    if (y_h == NULL) printf("We are computing curves on %s!\n",location.c_str());
    else printf("We are computing planes on %s!\n",location.c_str());
    printf("--------------------------------------\n\n");

    // Allocate matrices on host
    host_malloc_2d(&M_h, n);
    // Check allocation
    if (M_h == NULL) {
        printf("Allocation of matrix failed on host!\n");
        return;
    }
    memset(*M_h,0,n*n*sizeof(double));

    host_malloc_2d(&M_inv_h, n);
    // Check allocation
    if (M_inv_h == NULL) {
        printf("Allocation of matrix failed on host!\n");
        return;
    }
    memset(*M_inv_h,0,n*n*sizeof(double));

    // Allocate vectors on host
    cudaMallocHost((void **) &p_h, n*sizeof(double));

    // Check allocation
    if (p_h == NULL ) {
        printf("Allocation of vector failed on host!\n");
        return;
    }
    
    if (device) {

        // Allocate matrices on device
        device_malloc_2d(&M_d, &M_log, n);
        device_malloc_2d(&M_inv_d, &M_inv_log, n);

        // Check allocation
        if (M_d == NULL || M_log == NULL || M_inv_d == NULL || M_inv_log == NULL) {
            printf("Allocation of matrices failed on device! %d\n",n);
            return;
        }

        // Allocate vectors on device
        cudaMalloc((void **) &x_d,     n*sizeof(double));
        if (y_h != NULL) cudaMalloc((void **) &y_d,     n*sizeof(double));
        cudaMalloc((void **) &p_d,     n*sizeof(double));
        cudaMalloc((void **) &hyper_d,     num*sizeof(double));

        // Check allocation
        if (x_d == NULL || !(y_h == NULL || (y_h != NULL && y_d != NULL)) || p_d == NULL) {
            printf("Allocation of vectors failed on device!\n");
            return;
        }

        // Send to device
        cudaMemcpy(x_d, x_h, n * sizeof(double), cudaMemcpyHostToDevice);
        if (y_h != NULL) cudaMemcpy(y_d, y_h, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(M_log, *M_h, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(hyper_d, hyper_h, num * sizeof(double), cudaMemcpyHostToDevice);
        
    }

   
}

GaussianProcess::GaussianProcess(double* x, double* y, int n, double* hyper, int num, int dim, int dev, int type_covfunc) {

    if (dim == 1) {
        y = NULL;
    }

    // Save input parameters
    this->x_h       = x;
    this->y_h       = y;
    this->n         = n;
    this->hyper_h   = hyper;
    this->num       = num;
    this->dim       = dim;
    this->type_covfunc = type_covfunc;

    // Check if device is available
    int temp;
    cudaError_t cudaSuccess =  cudaGetDeviceCount(&temp);
    //device = (!cudaSuccess && temp > 0) ? true : false;
    //printf("devices = %d\n",cudaSuccess);

    device = (!cudaSuccess && temp > 0 && dev == 1) ? true : false;
    string location = device ? "device" : "host";

    printf("--------------------------------------\n");
    if (y_h == NULL) printf("We are computing curves on %s!\n",location.c_str());
    else printf("We are computing planes on %s!\n",location.c_str());
    printf("--------------------------------------\n\n");

    // Allocate matrices on host
    host_malloc_2d(&M_h, n);
    // Check allocation
    if (M_h == NULL) {
        printf("Allocation of matrix failed on host!\n");
        return;
    }
    memset(*M_h,0,n*n*sizeof(double));

    // Allocate vectors on host
    cudaMallocHost((void **) &p_h, n*sizeof(double));

    // Check allocation
    if (p_h == NULL ) {
        printf("Allocation of vector failed on host!\n");
        return;
    }
    
    if (device) {

        // Allocate matrices on device
        device_malloc_2d(&M_d, &M_log, n);

        // Check allocation
        if (M_d == NULL || M_log == NULL) {
            printf("Allocation of matrices failed on device! %d\n",n);
            return;
        }

        // Allocate vectors on device
        cudaMalloc((void **) &x_d,     n*sizeof(double));
        if (y_h != NULL) cudaMalloc((void **) &y_d,     n*sizeof(double));
        cudaMalloc((void **) &p_d,     n*sizeof(double));
        cudaMalloc((void **) &hyper_d,     num*sizeof(double));

        // Check allocation
        if (x_d == NULL || !(y_h == NULL || (y_h != NULL && y_d != NULL)) || p_d == NULL) {
            printf("Allocation of vectors failed on device!\n");
            return;
        }

        // Send to device
        cudaMemcpy(x_d, x_h, n * sizeof(double), cudaMemcpyHostToDevice);
        if (y_h != NULL) cudaMemcpy(y_d, y_h, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(M_log, *M_h, n * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(hyper_d, hyper_h, num * sizeof(double), cudaMemcpyHostToDevice);
        
    }

}

void GaussianProcess::free() {
    cudaError_t err;

    if (!device) {
        host_free_2d(M_h);
        host_free_2d(M_inv_h);
    }
    err = cudaFreeHost(x_h);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free memory on host: " << cudaGetErrorString(err) << std::endl;
    }
    if (dim == 2) {
        err = cudaFreeHost(y_h);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on host: " << cudaGetErrorString(err) << std::endl;
        }
    }
    err = cudaFreeHost(p_h);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free memory on host: " << cudaGetErrorString(err) << std::endl;
    }

    if (device) {
        device_free_2d(M_d,M_log);
        device_free_2d(M_inv_d,M_inv_log);
        err = cudaFree(x_d);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on device: " << cudaGetErrorString(err) << std::endl;
        }
        if (dim == 2) {
            err = cudaFree(y_d);
            if (err != cudaSuccess) {
                std::cerr << "Failed to free memory on device: " << cudaGetErrorString(err) << std::endl;
            }
        }

        err = cudaFree(p_d);
        if (err != cudaSuccess) {
            std::cerr << "Failed to free memory on device: " << cudaGetErrorString(err) << std::endl;
        }
    }

    printf("Gaussian Process freed!\n");

}

}