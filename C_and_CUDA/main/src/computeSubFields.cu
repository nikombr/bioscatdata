#include <stdlib.h>
#include <stdio.h>
//#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <omp.h>
//#include <cblas.h>
#include <math.h>
#include "../lib/Segment.h"
#include "../lib/BioScat.h"
#include "../lib/RealMatrix.h"
extern "C" {
using namespace std;

__global__ void combinePolarisationKernel(ComplexMatrix combined, ComplexMatrix P1, ComplexMatrix P2, int rows, double cosBeta, double sinBeta) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < rows) {
        combined.setDeviceRealValue(i, P1.getDeviceRealValue(i)*cosBeta + P2.getDeviceRealValue(i)*sinBeta);
        combined.setDeviceImagValue(i, P1.getDeviceImagValue(i)*cosBeta + P2.getDeviceImagValue(i)*sinBeta);
    }
}

void combinePolarisation(Field * pol, Field combined, double beta, bool deviceComputation) {

    int rows = combined.x.rows;
    double cosBeta = cos(beta);
    double sinBeta = sin(beta);
    
    if (deviceComputation) {

        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((rows + dimBlock.x - 1)/dimBlock.x);

        combinePolarisationKernel<<<dimGrid, dimBlock>>>(combined.x, pol[0].x, pol[1].x, rows, cosBeta, sinBeta);
        combinePolarisationKernel<<<dimGrid, dimBlock>>>(combined.y, pol[0].y, pol[1].y, rows, cosBeta, sinBeta);
        combinePolarisationKernel<<<dimGrid, dimBlock>>>(combined.z, pol[0].z, pol[1].z, rows, cosBeta, sinBeta);
        
        cudaDeviceSynchronize();
        //combined.toHost();

    }
    else {
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {           
            combined.x.setHostRealValue(i, pol[0].x.getHostRealValue(i)*cosBeta + pol[1].x.getHostRealValue(i)*sinBeta);
            combined.y.setHostRealValue(i, pol[0].y.getHostRealValue(i)*cosBeta + pol[1].y.getHostRealValue(i)*sinBeta);
            combined.z.setHostRealValue(i, pol[0].z.getHostRealValue(i)*cosBeta + pol[1].z.getHostRealValue(i)*sinBeta);
            combined.x.setHostImagValue(i, pol[0].x.getHostImagValue(i)*cosBeta + pol[1].x.getHostImagValue(i)*sinBeta);
            combined.y.setHostImagValue(i, pol[0].y.getHostImagValue(i)*cosBeta + pol[1].y.getHostImagValue(i)*sinBeta);
            combined.z.setHostImagValue(i, pol[0].z.getHostImagValue(i)*cosBeta + pol[1].z.getHostImagValue(i)*sinBeta);  
        }
    }

}

__global__ void computeScatteredSubFieldsKernelCombine(ComplexMatrix field, ComplexMatrix subField, int rows) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < rows) {
        field.setDeviceRealValue(k, field.getDeviceRealValue(k) + subField.getDeviceRealValue(k));
        field.setDeviceImagValue(k, field.getDeviceImagValue(k) + subField.getDeviceImagValue(k));
    }
}

void BioScat::computeScatteredSubFields() {

    int n = x_obs.rows;

    for (int i = 0; i < num_segments; i++) {
        
        double start_inner = omp_get_wtime();
        
        segments[i].computeScatteredFieldMatrices(x_obs, y_obs, false);

        segments[i].computeScatteredSubFields();

        //segments[i].freeScatteredFields();

        double end_inner = omp_get_wtime();

        if (printOutput) printf("\nIt took %.4e seconds to compute the scattered field matrices for segment %d.\n\n",end_inner - start_inner, i + 1);
        
    }

    double start = omp_get_wtime();
    
    if (true) {
     
        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);

        if (polarisation == 1) {
            
            //E_scat_pol[0].setDeviceZero();
            //H_scat_pol[0].setDeviceZero();
            
            for (int i = 0; i < num_segments; i++) {
                computeScatteredSubFieldsKernelCombine<<<dimGrid, dimBlock>>>(E_scat_pol[0].z, segments[i].E_scat.z, n);
                computeScatteredSubFieldsKernelCombine<<<dimGrid, dimBlock>>>(H_scat_pol[0].x, segments[i].H_scat.x, n);
                computeScatteredSubFieldsKernelCombine<<<dimGrid, dimBlock>>>(H_scat_pol[0].y, segments[i].H_scat.y, n);
                cudaDeviceSynchronize();
            }
           
            //E_scat_pol[0].toHost();
            //H_scat_pol[0].toHost();
        }
        else if (polarisation == 2) {
            //E_scat_pol[1].setDeviceZero();
            //H_scat_pol[1].setDeviceZero();
            for (int i = 0; i < num_segments; i++) {
                computeScatteredSubFieldsKernelCombine<<<dimGrid, dimBlock>>>(H_scat_pol[1].z, segments[i].H_scat.z, n);
                computeScatteredSubFieldsKernelCombine<<<dimGrid, dimBlock>>>(E_scat_pol[1].x, segments[i].E_scat.x, n);
                computeScatteredSubFieldsKernelCombine<<<dimGrid, dimBlock>>>(E_scat_pol[1].y, segments[i].E_scat.y, n);
                cudaDeviceSynchronize();
            }
            //E_scat_pol[1].toHost();
            //H_scat_pol[1].toHost();
        }
        

    }
    else {
        if (polarisation == 1) {
            for (int k = 0; k < n; k++) {
                double Ez_real, Ez_imag, Hx_real, Hx_imag, Hy_real, Hy_imag, C_real, C_imag;
                Ez_real = 0.0; Hx_real = 0.0; Hy_real = 0.0;
                Ez_imag = 0.0; Hx_imag = 0.0; Hy_imag = 0.0;
                for (int i = 0; i < num_segments; i++) {
                    // Get real values
                    Ez_real += segments[i].E_scat.z.getHostRealValue(k);
                    Hx_real += segments[i].H_scat.x.getHostRealValue(k);
                    Hy_real += segments[i].H_scat.y.getHostRealValue(k);

                    // Get imagninary values
                    Ez_imag += segments[i].E_scat.z.getHostImagValue(k);
                    Hx_imag += segments[i].H_scat.x.getHostImagValue(k);
                    Hy_imag += segments[i].H_scat.y.getHostImagValue(k);
                    
                }

                E_scat_pol[0].z.setHostRealValue(k, Ez_real);
                H_scat_pol[0].x.setHostRealValue(k, Hx_real);
                H_scat_pol[0].y.setHostRealValue(k, Hy_real);
                E_scat_pol[0].z.setHostImagValue(k, Ez_imag);
                H_scat_pol[0].x.setHostImagValue(k, Hx_imag);
                H_scat_pol[0].y.setHostImagValue(k, Hy_imag);
            }
            
        }
        else if (polarisation == 2) {
            for (int k = 0; k < n; k++) {
                double Hz_real, Hz_imag, Ex_real, Ex_imag, Ey_real, Ey_imag, C_real, C_imag;
                Hz_real = 0.0; Ex_real = 0.0; Ey_real = 0.0;
                Hz_imag = 0.0; Ex_imag = 0.0; Ey_imag = 0.0;
                for (int i = 0; i < num_segments; i++) {

                        // Computing real values
                        Hz_real += segments[i].H_scat.z.getHostRealValue(k);
                        Ex_real += segments[i].E_scat.x.getHostRealValue(k);
                        Ey_real += segments[i].E_scat.y.getHostRealValue(k);

                        // Computing complex values
                        Hz_imag += segments[i].H_scat.z.getHostImagValue(k);
                        Ex_imag += segments[i].E_scat.x.getHostImagValue(k);
                        Ey_imag += segments[i].E_scat.y.getHostImagValue(k);
                
                }

                H_scat_pol[1].z.setHostRealValue(k, Hz_real);
                E_scat_pol[1].x.setHostRealValue(k, Ex_real);
                E_scat_pol[1].y.setHostRealValue(k, Ey_real);
                H_scat_pol[1].z.setHostImagValue(k, Hz_imag);
                E_scat_pol[1].x.setHostImagValue(k, Ex_imag);
                E_scat_pol[1].y.setHostImagValue(k, Ey_imag);
            }  
        }
    }

    double end = omp_get_wtime();
    if (printOutput) printf("\nIt took %.4e seconds to compute the combined scattered fields in the observation points.\n\n",end-start);
    for (int i = 0; i < num_segments; i++) {
        //segments[i].freeScatteredSubFields();
    }

}


__global__ void copyRealKernel(ComplexMatrix output, ComplexMatrix input, int n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        output.setDeviceRealValue(j,input.getDeviceRealValue(j));
    }
}

__global__ void copyImagKernel(ComplexMatrix output, ComplexMatrix input, int n) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if (j < n) {
        output.setDeviceImagValue(j,input.getDeviceImagValue(j));
    }
}


void BioScat::computeIncidentSubFields() {
    int n = x_obs.rows;
    double val;
    double start = omp_get_wtime();
    segments[0].computeIncidentFieldVectors(y_obs);
    if (true) {
        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);
        if (polarisation == 1) {
            copyRealKernel<<<dimGrid, dimBlock>>>(E_inc_pol[0].z, segments[0].E_inc_vector.z, n);
            copyImagKernel<<<dimGrid, dimBlock>>>(E_inc_pol[0].z, segments[0].E_inc_vector.z, n);
            copyRealKernel<<<dimGrid, dimBlock>>>(H_inc_pol[0].x, segments[0].H_inc_vector.x, n);
            copyImagKernel<<<dimGrid, dimBlock>>>(H_inc_pol[0].x, segments[0].H_inc_vector.x, n);
        }
        else if (polarisation == 2) {
            copyRealKernel<<<dimGrid, dimBlock>>>(H_inc_pol[1].z, segments[0].H_inc_vector.z, n);
            copyImagKernel<<<dimGrid, dimBlock>>>(H_inc_pol[1].z, segments[0].H_inc_vector.z, n);
            copyRealKernel<<<dimGrid, dimBlock>>>(E_inc_pol[1].x, segments[0].E_inc_vector.x, n);
            copyImagKernel<<<dimGrid, dimBlock>>>(E_inc_pol[1].x, segments[0].E_inc_vector.x, n);
        }

        cudaDeviceSynchronize();

        //E_inc_pol[0].toHost();
        //H_inc_pol[0].toHost();
        //E_inc_pol[1].toHost();
        //H_inc_pol[1].toHost();
    }
    else {
        if (polarisation == 1) {
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) E_inc_pol[0].z.setHostRealValue(k, segments[0].E_inc_vector.z.getHostRealValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) E_inc_pol[0].z.setHostImagValue(k, segments[0].E_inc_vector.z.getHostImagValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) H_inc_pol[0].x.setHostRealValue(k, segments[0].H_inc_vector.x.getHostRealValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) H_inc_pol[0].x.setHostImagValue(k, segments[0].H_inc_vector.x.getHostImagValue(k));
        }
        else if (polarisation == 2) {
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) H_inc_pol[1].z.setHostRealValue(k, segments[0].H_inc_vector.z.getHostRealValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) H_inc_pol[1].z.setHostImagValue(k, segments[0].H_inc_vector.z.getHostImagValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) E_inc_pol[1].x.setHostRealValue(k, segments[0].E_inc_vector.x.getHostRealValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) E_inc_pol[1].x.setHostImagValue(k, segments[0].E_inc_vector.x.getHostImagValue(k));
        }
    }
    double end = omp_get_wtime();
    if (printOutput) printf("\nIt took %.4e seconds to compute the combined incident fields in the observation points.\n\n",end-start);
    
    //segments[0].freeIncidentFields();
    
}

void BioScat::computeReflectedSubFields() {

    int n = x_obs.rows;
    double val;
    double start = omp_get_wtime();
    segments[0].computeReflectedFieldVectors(y_obs);
    if (true) {
        // Blocks and threads
        dim3 dimBlock(256);
        dim3 dimGrid((n + dimBlock.x - 1)/dimBlock.x);
        if (polarisation == 1) {
            copyRealKernel<<<dimGrid, dimBlock>>>(E_ref_pol[0].z, segments[0].E_ref_vector.z, n);
            copyImagKernel<<<dimGrid, dimBlock>>>(E_ref_pol[0].z, segments[0].E_ref_vector.z, n);
            copyRealKernel<<<dimGrid, dimBlock>>>(H_ref_pol[0].x, segments[0].H_ref_vector.x, n);
            copyImagKernel<<<dimGrid, dimBlock>>>(H_ref_pol[0].x, segments[0].H_ref_vector.x, n);
        }
        else if (polarisation == 2) {
            copyRealKernel<<<dimGrid, dimBlock>>>(H_ref_pol[1].z, segments[0].H_ref_vector.z, n);
            copyImagKernel<<<dimGrid, dimBlock>>>(H_ref_pol[1].z, segments[0].H_ref_vector.z, n);
            copyRealKernel<<<dimGrid, dimBlock>>>(E_ref_pol[1].x, segments[0].E_ref_vector.x, n);
            copyImagKernel<<<dimGrid, dimBlock>>>(E_ref_pol[1].x, segments[0].E_ref_vector.x, n);
        }

        cudaDeviceSynchronize();

        //E_ref_pol[0].toHost();
        //H_ref_pol[0].toHost();
        //E_ref_pol[1].toHost();
        //H_ref_pol[1].toHost();
    }
    else {
        
        if (polarisation == 1) {
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) E_ref_pol[0].z.setHostRealValue(k, segments[0].E_ref_vector.z.getHostRealValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) E_ref_pol[0].z.setHostImagValue(k, segments[0].E_ref_vector.z.getHostImagValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) H_ref_pol[0].x.setHostRealValue(k, segments[0].H_ref_vector.x.getHostRealValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) H_ref_pol[0].x.setHostImagValue(k, segments[0].H_ref_vector.x.getHostImagValue(k));
        }
        else if (polarisation == 2) {
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) H_ref_pol[1].z.setHostRealValue(k, segments[0].H_ref_vector.z.getHostRealValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) H_ref_pol[1].z.setHostImagValue(k, segments[0].H_ref_vector.z.getHostImagValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) E_ref_pol[1].x.setHostRealValue(k, segments[0].E_ref_vector.x.getHostRealValue(k));
            //#pragma omp parallel for
            for (int k = 0; k < n; k++) E_ref_pol[1].x.setHostImagValue(k, segments[0].E_ref_vector.x.getHostImagValue(k));
        }
    }
    double end = omp_get_wtime();
    if (printOutput) printf("\nIt took %.4e seconds to compute the combined reflected fields in the observation points.\n\n",end-start);
    
    //segments[0].freeReflectedFields();
    
}



}