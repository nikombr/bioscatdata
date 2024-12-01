#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
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

void setupSystemMatrix_CPU(RealMatrix A, int n_test, int n_int, int n_ext, RealMatrix n_x, RealMatrix n_y, ComplexMatrix firstField_scat, ComplexMatrix firstField_int,  ComplexMatrix secondField_scat, ComplexMatrix secondField_int, ComplexMatrix thirdField_scat, ComplexMatrix thirdField_int) {
    bool host = true; 
    bool device = false;
    RealMatrix A_real = RealMatrix(2 * n_test, n_int + n_ext, host, device);
    RealMatrix A_imag = RealMatrix(2 * n_test, n_int + n_ext, host, device);
    double val;

    for (int r = 0; r < n_test; r++) {
        for (int c = 0; c < n_int; c++) {
            val =  firstField_scat.getHostRealValue(r, c);
            A_real.setHostValue(r, c, val);
            val =  firstField_scat.getHostImagValue(r, c);
            A_imag.setHostValue(r, c, val);
        }
        for (int c = 0; c < n_ext; c++) {
            val =  -firstField_int.getHostRealValue(r, c);
            A_real.setHostValue(r, c + n_int, val);
            val =  -firstField_int.getHostImagValue(r, c);
            A_imag.setHostValue(r, c + n_int, val);
        }
       
    }

    for (int r = 0; r < n_test; r++) {
        for (int c = 0; c < n_int; c++) {
            val  = 0;
            val += - n_y.getHostValue(r) * secondField_scat.getHostRealValue(r, c);
            val +=   n_x.getHostValue(r) * thirdField_scat.getHostRealValue(r, c);
            A_real.setHostValue(r + n_test, c, val);
            val  = 0;
            val += - n_y.getHostValue(r) * secondField_scat.getHostImagValue(r, c);
            val +=   n_x.getHostValue(r) * thirdField_scat.getHostImagValue(r, c);
            A_imag.setHostValue(r + n_test, c, val);
        }
        for (int c = 0; c < n_ext; c++) {
            val  = 0;
            val +=   n_y.getHostValue(r) * secondField_int.getHostRealValue(r, c);
            val += - n_x.getHostValue(r) * thirdField_int.getHostRealValue(r, c);
            A_real.setHostValue(r + n_test, c + n_int, val);
            val  = 0;
            val +=   n_y.getHostValue(r) * secondField_int.getHostImagValue(r, c);
            val += - n_x.getHostValue(r) * thirdField_int.getHostImagValue(r, c);
            A_imag.setHostValue(r + n_test, c + n_int, val);
        }
    }

    
    for (int r = 0; r < 2 * n_test; r++) {
        for (int c = 0; c < n_ext + n_int; c++) {
            A.setHostValue(r,              c,                   A_real.getHostValue(r,c));
            A.setHostValue(r,              c + n_ext + n_int, - A_imag.getHostValue(r,c));
            A.setHostValue(r + 2 * n_test, c,                   A_imag.getHostValue(r,c));
            A.setHostValue(r + 2 * n_test, c + n_ext + n_int,   A_real.getHostValue(r,c));
        }
    }

    A_real.free();
    A_imag.free();

}

__global__ void computeScatteringAmatrix(RealMatrix A_real, RealMatrix A_imag, int n_test, int n_int, RealMatrix n_x, RealMatrix n_y, ComplexMatrix firstField_scat, ComplexMatrix secondField_scat, ComplexMatrix thirdField_scat) {
    double val;
    //printf("in kernel\n");
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    if (r < n_test && c < n_int) {
        //printf("her1\n");
        val =  firstField_scat.getDeviceRealValue(r, c);
        A_real.setDeviceValue(r, c, val);
        val =  firstField_scat.getDeviceImagValue(r, c);
        A_imag.setDeviceValue(r, c, val);

        val  = 0;
        val += - n_y.getDeviceValue(r) * secondField_scat.getDeviceRealValue(r, c);
        val +=   n_x.getDeviceValue(r) * thirdField_scat.getDeviceRealValue(r, c);
        A_real.setDeviceValue(r + n_test, c, val);
        val  = 0;
        val += - n_y.getDeviceValue(r) * secondField_scat.getDeviceImagValue(r, c);
        val +=   n_x.getDeviceValue(r) * thirdField_scat.getDeviceImagValue(r, c);
        A_imag.setDeviceValue(r + n_test, c, val);
    }
}

__global__ void computeInteriorAmatrix(RealMatrix A_real, RealMatrix A_imag, int n_test, int n_int, int n_ext, RealMatrix n_x, RealMatrix n_y, ComplexMatrix firstField_int, ComplexMatrix secondField_int, ComplexMatrix thirdField_int) {
    double val;
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    int n = n_int + n_ext;
    if (r < n_test && c < n_ext) {

        val =  -firstField_int.getDeviceRealValue(r, c);
        A_real.setDeviceValue(r, c + n_int, val);
        val =  -firstField_int.getDeviceImagValue(r, c);
        A_imag.setDeviceValue(r, c + n_int, val);

        val  = 0;
        val +=   n_y.getDeviceValue(r) * secondField_int.getDeviceRealValue(r, c);
        val += - n_x.getDeviceValue(r) * thirdField_int.getDeviceRealValue(r, c);
        A_real.setDeviceValue(r + n_test, c + n_int, val);
        val  = 0;
        val +=   n_y.getDeviceValue(r) * secondField_int.getDeviceImagValue(r, c);
        val += - n_x.getDeviceValue(r) * thirdField_int.getDeviceImagValue(r, c);
        A_imag.setDeviceValue(r + n_test, c + n_int, val);

    }
}

__global__ void combineAmatrixLoop(RealMatrix A, RealMatrix A_real, RealMatrix A_imag, int n_test, int n_int,int n_ext) {
    //printf("combine\n");
    double val;
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    int c = threadIdx.y + blockIdx.y * blockDim.y;
    int n = n_int + n_ext;
    int nA = 2*n;
    if (r < 2*n_test && c < n_int + n_ext) {
        
        A.setDeviceValue(r,              c,                   A_real.getDeviceValue(r,c));
        A.setDeviceValue(r,              c + n_ext + n_int, - A_imag.getDeviceValue(r,c));
        A.setDeviceValue(r + 2 * n_test, c,                   A_imag.getDeviceValue(r,c));
        A.setDeviceValue(r + 2 * n_test, c + n_ext + n_int,   A_real.getDeviceValue(r,c));

    }
}

__global__ void testker(double * A_real, double * A_imag, int n_test, int n_int, int n_ext, double * n_x, double * n_y, double * firstField_scat_real, double * firstField_scat_imag, double * secondField_scat_real, double * secondField_scat_imag, double * thirdField_scat_real, double * thirdField_scat_imag) {
    printf("test\n");
}

void setupSystemMatrix_GPU(RealMatrix A, int n_test, int n_int, int n_ext, RealMatrix n_x, RealMatrix n_y, ComplexMatrix firstField_scat, ComplexMatrix firstField_int, ComplexMatrix secondField_scat, ComplexMatrix secondField_int, ComplexMatrix thirdField_scat, ComplexMatrix thirdField_int) {
    bool host = false;
    bool device = true;
    RealMatrix A_real = RealMatrix(2 * n_test, n_int + n_ext, host, device);
    RealMatrix A_imag = RealMatrix(2 * n_test, n_int + n_ext, host, device);

    // Blocks and threads
    dim3 dimBlock(32,32);
    dim3 dimGrid((n_test + dimBlock.x - 1)/dimBlock.x, (n_int + dimBlock.y - 1)/dimBlock.y);
   
    //testker<<<dimGrid, dimBlock>>>();
    cudaDeviceSynchronize();
    /*double * A_real_d  = A_real.getDevicePointer();
    double * A_imag_d  = A_imag.getDevicePointer();
    double * n_x_d     = n_x.getDevicePointer();
    double * n_y_d     = n_y.getDevicePointer();
    double * f1_real_d = firstField_scat.getDeviceRealPointer();
    double * f1_imag_d = firstField_scat.getDeviceImagPointer();
    double * f2_real_d = secondField_scat.getDeviceRealPointer();
    double * f2_imag_d = secondField_scat.getDeviceImagPointer();
    double * f3_real_d = thirdField_scat.getDeviceRealPointer();
    double * f3_imag_d = thirdField_scat.getDeviceImagPointer();
    if (f3_real_d == NULL) printf("this is null\n");
    if (f3_imag_d == NULL) printf("this is null\n");
    printf("hmm\n");
    cudaDeviceSynchronize();*/
    computeScatteringAmatrix<<<dimGrid, dimBlock>>>(A_real, A_imag, n_test, n_int, n_x,  n_y, \
                                                    firstField_scat, secondField_scat, thirdField_scat);
    cudaCheckError(); 
    cudaDeviceSynchronize();
    // Blocks and threads
    dimGrid.y = (n_ext  + dimBlock.y - 1)/dimBlock.y;
   
    computeInteriorAmatrix<<<dimGrid, dimBlock>>>(A_real, A_imag, n_test, n_int, n_ext, n_x,  n_y, \
                                     firstField_int, secondField_int, thirdField_int);
    cudaCheckError(); 
    // Synchronize threads
    cudaDeviceSynchronize();

    // Blocks and threads
    dimGrid.x = (2*n_test + dimBlock.x - 1)/dimBlock.x;
    dimGrid.y = (n_int + n_ext + dimBlock.y - 1)/dimBlock.y;
 
    combineAmatrixLoop<<< dimGrid, dimBlock>>>(A, A_real, A_imag, n_test, n_int, n_ext);
    cudaCheckError(); 
    // Synchronize threads
    cudaDeviceSynchronize();

    A_real.free();
    A_imag.free();

}

void Segment::setupSystemMatrix() {

    if (deviceComputation) {
        if (polarisation == 1) {
            setupSystemMatrix_GPU(A, n_test, n_int, n_ext, n_x, n_y, E_scat_matrix.z, E_int_matrix.z, H_scat_matrix.x, H_int_matrix.x, H_scat_matrix.y, H_int_matrix.y);
        }
        else if (polarisation == 2) {
            setupSystemMatrix_GPU(A, n_test, n_int, n_ext, n_x, n_y, H_scat_matrix.z, H_int_matrix.z, E_scat_matrix.x, E_int_matrix.x, E_scat_matrix.y, E_int_matrix.y);
        }
        else {
            printf("You have to choose either 1 or 2 as the polarisation!\n");
            return;
        }
    }
    else {
         if (polarisation == 1) {
            setupSystemMatrix_CPU(A, n_test, n_int, n_ext, n_x, n_y, E_scat_matrix.z, E_int_matrix.z, H_scat_matrix.x, H_int_matrix.x, H_scat_matrix.y, H_int_matrix.y);
        }
        else if (polarisation == 2) {
            setupSystemMatrix_CPU(A, n_test, n_int, n_ext, n_x, n_y, H_scat_matrix.z, H_int_matrix.z, E_scat_matrix.x, E_int_matrix.x, E_scat_matrix.y, E_int_matrix.y);
        }
        else {
            printf("You have to choose either 1 or 2 as the polarisation!\n");
            return;
        }

    }



    if (n_test < 200 && !deviceComputation) {
    FILE *file;
    char * filename;
    /*filename = "A_real_C.txt";
    file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    for (int r = 0; r < 2*n_test; r++) {
        for (int c = 0; c < n_int + n_ext; c++) {
            fprintf(file, "%e ", A_real.getHostValue(r,c));
        }
        fprintf(file, "\n");
    }
    fclose(file);
    filename = "A_imag_C.txt";
    file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    for (int r = 0; r < 2*n_test; r++) {
        for (int c = 0; c < n_int + n_ext; c++) {
            fprintf(file, "%e ", A_imag.getHostValue(r,c));
        }
        fprintf(file, "\n");
    }
    fclose(file);*/

    filename = "Abig_C.txt";
    file = fopen(filename, "w");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    for (int r = 0; r < A.rows; r++) {
        for (int c = 0; c < A.cols; c++) {
            fprintf(file, "%e ", A.getHostValue(r,c));
        }
        fprintf(file, "\n");
    }
    fclose(file);

    

    }

    

    return;
}


}