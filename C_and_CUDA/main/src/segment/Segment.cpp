#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include "../../lib/cuSolver.h"
extern "C" {
#include "../../lib/Segment.h"
#include "../../lib/RealMatrix.h"
#include "../../lib/ComplexMatrix.h"
using namespace std;

Segment::Segment() {
    constants = Constants();
}

void Segment::newWavelength(double lambda) {
    constants.newWavelength(lambda);
}

void Segment::freeScatteredFields() {
    E_scat_matrix.free();
    H_scat_matrix.free();
}

void Segment::freeScatteredSubFields() {
    E_scat.free();
    H_scat.free();
}

void Segment::freeIncidentFields() {
    E_inc_vector.free();
    H_inc_vector.free();
}

void Segment::freeReflectedFields() {
    E_ref_vector.free();
    H_ref_vector.free();
}

void Segment::freeInteriorFields() {
    E_int_matrix.free(); 
    H_int_matrix.free(); 
    
}
    
void Segment::free() {
    // Frees all allocated arrays

    x_int.free();
    y_int.free();
    x_ext.free();
    y_ext.free();
    x_test.free();
    y_test.free();
    n_x.free();
    n_y.free();
    C.free();
    D.free();
    A.free();
    b.free();
    freeScatteredFields();
    freeScatteredSubFields();
    freeIncidentFields();
    freeInteriorFields();
    freeReflectedFields();
    cudaFree(A_T_d);
    cudaFree(x_d);
    cusolverDnDestroy(handle);
    
}

void Segment::allocate() {
    // Allocates arrays
    bool host = !deviceComputation;
    bool device = deviceComputation;
    int n = std::max(n_obs, n_test);
    if (printOutput) printf("Allocate: (n_test, n_int, n_ext, n_obs, n) = (%d, %d, %d, %d, %d)\n", n_test, n_int, n_ext, n_obs, n);
    x_int         = RealMatrix(n_int,                         host, device);
    y_int         = RealMatrix(n_int,                         host, device);
    x_ext         = RealMatrix(n_ext,                         host, device);
    y_ext         = RealMatrix(n_ext,                         host, device);
    x_test        = RealMatrix(n_test,                        host, device);
    y_test        = RealMatrix(n_test,                        host, device);
    n_x           = RealMatrix(n_test,                        host, device);
    n_y           = RealMatrix(n_test,                        host, device);
    C             = ComplexMatrix(n_int,                      host, device);
    D             = ComplexMatrix(n_ext,                      host, device);
    A             = RealMatrix(4 * n_test, 2*(n_ext + n_int), host, device);
    b             = RealMatrix(4 * n_test,                    host, device);
    E_scat_matrix = Field(n, n_int,                           host, device);
    H_scat_matrix = Field(n, n_int,                           host, device);
    E_scat        = Field(n,                                  host, device);
    H_scat        = Field(n,                                  host, device);
    F             = ComplexMatrix(n,                             host, device);
    E_int_matrix  = Field(n, n_ext,                           host, device);
    H_int_matrix  = Field(n, n_ext,                           host, device);
    E_inc_vector  = Field(n,                                  host, device);
    H_inc_vector  = Field(n,                                  host, device);
    E_ref_vector  = Field(n,                                  host, device);
    H_ref_vector  = Field(n,                                  host, device);
    // Prepare for linear system
    cusolverDnCreate(&handle);
    cudaMalloc((void **) &A_T_d,    A.rows * A.cols * sizeof(double));
    cudaMalloc((void **) &x_d,      A.cols * sizeof(double));
    
    


    

}

}