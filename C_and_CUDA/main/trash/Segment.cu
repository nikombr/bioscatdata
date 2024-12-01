// Part of segment that applies both to 2D and 3D
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
extern "C" {
#include "../lib/2D/Segment.h"
#include "../lib/RealMatrix.h"
using namespace std;

void Segment::computeIncidentFieldVectors(RealMatrix y) {
    
    int rows = y.rows;

    bool Ex_bool = scenario == 1 ? false : true;
    bool Ez_bool = scenario == 1 ? true  : false;
    bool Hx_bool = scenario == 1 ? true  : false;
    bool Hz_bool = scenario == 1 ? false : true;
    bool Ey_bool = false;
    bool Hy_bool = false;

    E_inc_vector = Field(rows, Ex_bool, Ey_bool, Ez_bool);
    H_inc_vector = Field(rows, Hx_bool, Hy_bool, Hz_bool);

    if (scenario == 1) {
        for (int j = 0; j < rows; j++) E_inc_vector.z.setHostRealValue(j,                     cos(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) E_inc_vector.z.setHostImagValue(j,                     sin(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) H_inc_vector.x.setHostRealValue(j, -1/constants.eta0 * cos(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) H_inc_vector.x.setHostImagValue(j, -1/constants.eta0 * sin(constants.k0 * y.getHostValue(j)));
    }
    else {
        for (int j = 0; j < rows; j++) E_inc_vector.x.setHostRealValue(j,                    cos(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) E_inc_vector.x.setHostImagValue(j,                    sin(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) H_inc_vector.z.setHostRealValue(j, 1/constants.eta0 * cos(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) H_inc_vector.z.setHostImagValue(j, 1/constants.eta0 * sin(constants.k0 * y.getHostValue(j)));
    }


}

void Segment::computeReflectedFieldVectors(RealMatrix y) {
    
    int rows = y.rows;

    bool Ex_bool = scenario == 1 ? false : true;
    bool Ez_bool = scenario == 1 ? true  : false;
    bool Hx_bool = scenario == 1 ? true  : false;
    bool Hz_bool = scenario == 1 ? false : true;
    bool Ey_bool = false;
    bool Hy_bool = false;

    E_ref_vector = Field(rows, Ex_bool, Ey_bool, Ez_bool);
    H_ref_vector = Field(rows, Hx_bool, Hy_bool, Hz_bool);

    if (scenario == 1) {
        for (int j = 0; j < rows; j++) E_ref_vector.z.setHostRealValue(j,                     constants.Gamma_ref * cos(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) E_ref_vector.z.setHostImagValue(j,                    -constants.Gamma_ref * sin(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) H_ref_vector.x.setHostRealValue(j,  1/constants.eta0 * constants.Gamma_ref * cos(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) H_ref_vector.x.setHostImagValue(j, -1/constants.eta0 * constants.Gamma_ref * sin(constants.k0 * y.getHostValue(j)));
    }
    else {
        for (int j = 0; j < rows; j++) E_ref_vector.x.setHostRealValue(j,                     constants.Gamma_ref * cos(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) E_ref_vector.x.setHostImagValue(j,                    -constants.Gamma_ref * sin(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) H_ref_vector.z.setHostRealValue(j, -1/constants.eta0 * constants.Gamma_ref * cos(constants.k0 * y.getHostValue(j)));
        for (int j = 0; j < rows; j++) H_ref_vector.z.setHostImagValue(j,  1/constants.eta0 * constants.Gamma_ref * sin(constants.k0 * y.getHostValue(j)));
    }


}
  

void Segment::setupRightHandSide() {
    // Setup right-hand side in linear system
    b_imag = RealMatrix(2 * num_test_points);
    b_real = RealMatrix(2 * num_test_points);

    double val, shift;
    //int imagShift = 2*num_test_points;
    ComplexMatrix *firstField_inc, * firstField_ref, *secondField_inc,*secondField_ref;
    //printf("scenario = %d\n",scenario);
    //printf("%d %\n",scenario == 1, scenario == 2);
    if (scenario == 1) {
        //printf("SCENARIO 1");
        firstField_inc = &E_inc_vector.z;
        firstField_ref = &E_ref_vector.z;
        secondField_inc = &H_inc_vector.x;
        secondField_ref = &H_ref_vector.x;
    }
    else if (scenario == 2) {
        firstField_inc = &H_inc_vector.x;
        firstField_ref = &H_ref_vector.x;
        secondField_inc = &E_inc_vector.z;
        secondField_ref = &E_ref_vector.z;
    }
    else {
        printf("You have to choose either 1 or 2 as the scenario!\n");
        return;
    }

    for (int j = 0; j < num_test_points; j++) {
        val =  - firstField_inc->getHostRealValue(j) - firstField_ref->getHostRealValue(j);
        b_real.setHostValue(j, val);
        val =  - firstField_inc->getHostImagValue(j) - firstField_ref->getHostImagValue(j);
        b_imag.setHostValue(j, val);
    }

    shift = num_test_points;
    for (int j = 0; j < n_top; j++) {
        val  = secondField_inc->getHostRealValue(j) + secondField_ref->getHostRealValue(j);
        val *= n_y.getHostValue(j);
        b_real.setHostValue(j + shift, val);
        val  = secondField_inc->getHostImagValue(j) + secondField_ref->getHostImagValue(j);
        val *= n_y.getHostValue(j);
        b_imag.setHostValue(j + shift, val);
    }

    shift += n_top;
    for(int j = 0; j < n_right; j++) {
        val = 0.0;
        b_real.setHostValue(j + shift, val);
        b_imag.setHostValue(j + shift, val);
    }

    shift += n_right;
    for(int j = 0; j < n_bottom; j++) {
        val  = secondField_inc->getHostRealValue(j + n_top + n_right) + secondField_ref->getHostRealValue(j + n_top + n_right);
        b_real.setHostValue(j + shift, val);
        val  = secondField_inc->getHostImagValue(j + n_top + n_right) + secondField_ref->getHostImagValue(j + n_top + n_right);
        val = 555;
        b_imag.setHostValue(j + shift, val);
    }

    shift += n_bottom;
    for(int j = 0; j < n_left; j++) {
        val = 0.0;
        b_real.setHostValue(j + shift, val);
        b_imag.setHostValue(j + shift, val);
    }
    /*printf("b:\n");
    for (int j = 0; j < 2*num_test_points; j++) {
        printf("%e\t + i(%e)\n",b_real.getHostValue(j),b_imag.getHostValue(j));
    }*/
    
    return;
 
}

void Segment::setupSystemMatrix() {
    A = RealMatrix(4 * num_test_points, 2*(n_ext + n_int));
    for (int r = 0; r < 2 * num_test_points; r++) {
        for (int c = 0; c < n_ext + n_int; c++) {
            A.setHostValue(r,                       c,                   A_real.getHostValue(r,c));
            A.setHostValue(r,                       c + n_ext + n_int, - A_imag.getHostValue(r,c));
            A.setHostValue(r + 2 * num_test_points, c,                   A_imag.getHostValue(r,c));
            A.setHostValue(r + 2 * num_test_points, c + n_ext + n_int,   A_real.getHostValue(r,c));
        }
    }
}

// LAPACK routine for solving linear system
void dgels_(const char * trans, const int * m, const int * n, const int * nrhs, double * A, const int * lda, double * B,  const int * ldb, double * work, int * lwork,int * info);
//#include <lapacke.h>
void Segment::solveLinearSystem() {

    setupSystemMatrix();
    
    b = RealMatrix(4 * num_test_points);

    for (int r = 0; r < 2*num_test_points; r++) {
        b.setHostValue(r,                     b_real.getHostValue(r));
        b.setHostValue(r + 2*num_test_points, b_imag.getHostValue(r));
    }

    /*A = RealMatrix(2*(n_ext + n_int),4 * num_test_points);
    b = RealMatrix(4 * num_test_points);

    for (int r = 0; r < 2*num_test_points; r++) {
        for (int c = 0; c < n_ext + n_int; c++) {
            A.setHostValue(c,                   r,                     A_real.getHostValue(r,c));
            A.setHostValue(c + n_ext + n_int, r,                      - A_imag.getHostValue(r,c));
            A.setHostValue(c,         r + 2*num_test_points,           A_imag.getHostValue(r,c));
            A.setHostValue(c + n_ext + n_int,r + 2*num_test_points,    A_real.getHostValue(r,c));
        }
    }
    for (int r = 0; r < 2*num_test_points; r++) {
        b.setHostValue(r,                     b_real.getHostValue(r));
        b.setHostValue(r + 2*num_test_points, b_imag.getHostValue(r));
    }*/
    /*
    double Atest[12] = {1, 2,  3,  4,
                        5, 6,  7,  8,
                        9, 10, 11, 12};
    */
    /*
    double Atest[12] = {1,5,9,
                        2,6,10}; 
    */
    double Atest[12] = {1, 2, 
                        5, 6, 
                        9, 10}; 

    double btest[4];
    btest[0] = 5;
    btest[1] = 17;
    btest[2] = 29;
    int rows = 3;
    int cols = 2;

    char trans = 'T';
    int m = A.cols;//4 * num_test_points;
    int n = A.rows;//2*(n_ext + n_int);
    //printf("%d %d\n",m,n);
    int nrhs = 1; 
    int lda = m;
    int ldb = std::max(m, n);
    int info;
    double work_query;
    int lwork = -1;
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            //printf("%.6e\t",A.getHostValue(j,i));
        }
        //printf("\n");
    }


    dgels_(&trans, &m, &n, &nrhs, A.getHostPointer(), &lda, b.getHostPointer(), &ldb, &work_query, &lwork, &info);
    
    lwork = (int)work_query;
    double *work = (double*)malloc(lwork * sizeof(double));

    /*printf("\n\n");
    printf("A:\n");
    for (int j = 0; j < 2*num_test_points; j++) {
        printf("%f\t + i(%f)\n",A_real.getHostValue(j,0),A_imag.getHostValue(j,0));
    }*/

    printf("Solving linear system now.\n");
    dgels_(&trans, &m, &n, &nrhs, A.getHostPointer(), &lda, b.getHostPointer(), &ldb, work, &lwork, &info);
    //printf("HEJ\n");
    //info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, nrhs, A, lda, B, ldb);
    if (info != 0) {
        printf("An error occurred in solving: %d\n", info);
    }

    C = ComplexMatrix(n_int);
    D = ComplexMatrix(n_ext);

    for (int i = 0; i < n_int; i++) {
        C.setHostRealValue(i,b.getHostValue(i));
        C.setHostImagValue(i,b.getHostValue(i + n_ext + n_int));
    }
    
    for (int i = 0; i < n_ext; i++) {
        D.setHostRealValue(i,b.getHostValue(i + n_int));
        D.setHostImagValue(i,b.getHostValue(i + n_ext + 2*n_int));
    }
    /*printf("b:\n");

    for (int j = 0; j < 4; j++) {
        printf("%f\n",btest[j]);
    }

    printf("C:\n");

    for (int j = 0; j < n_int; j++) {
        printf("%e\t + i(%e)\n",b.getHostValue(j),b.getHostValue(j+n_ext+n_int));
    }

    printf("D:\n");

    for (int j = n_int; j < n_int + n_ext; j++) {
        printf("%e\t + i(%e)\n",b.getHostValue(j),b.getHostValue(j+n_ext+n_int));
    }*/
    
 
}


}