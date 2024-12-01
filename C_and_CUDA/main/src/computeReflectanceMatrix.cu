#include <stdlib.h>
#include <stdio.h>
//#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <stdexcept>
#include "../lib/ComplexMatrix.h"
#include "../lib/BioScat.h"
#include "../lib/Segment.h"
extern "C" {


__global__ void storeReflectanceKernel(RealMatrix reflectance_output, RealMatrix reflectance_input, int i, int j, int n) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k < n) {
        reflectance_output.setDeviceValue(k,j,i,reflectance_input.getDeviceValue(k));
    }
}

void showProgressBar(float progress) {
    // From chatgpt
    int barWidth = 100; // Width of the progress bar in characters

    std::cout << " ";
    int pos = barWidth * progress; // Position of the current progress in the bar
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "|";
        //else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << " " << int(progress * 100.0) << " %\r"; // Display percentage
    std::cout.flush();
}

void computeReflectanceMatrixGetData(BioScat bioscat, RealMatrix reflectance, double * lambdas, double * betas, int num_lambdas, int num_betas, int n_obs, bool deviceComputation)  {

    bioscat.getSegments();

    for (int i = 0; i < num_lambdas; i++) {
        double lambda = lambdas[i];
        bioscat.reset(); // Set fields to zero
        bioscat.prepareForward(lambda);
         double start_test = omp_get_wtime();
        for (int polarisation = 1; polarisation <= 2; polarisation++) {
            bioscat.forwardSolver(polarisation);
            bioscat.computeScatteredSubFields();
            //bioscat.computeReflectedSubFields();
            bioscat.computeIncidentSubFields();
        }

       
        for (int j = 0; j < num_betas; j++) {
            showProgressBar((i*num_betas + j) / ((double) num_lambdas*num_betas));
            double beta = betas[j];
            
            bioscat.computeScatteredFields(beta);
            //bioscat.computeReflectedFields(beta);
            bioscat.computeIncidentFields(beta);
            bioscat.computeReflectance();
            
            if (deviceComputation) {
                // Blocks and threads
                dim3 dimBlock(256);
                dim3 dimGrid((n_obs + dimBlock.x - 1)/dimBlock.x);
                storeReflectanceKernel<<<dimGrid, dimBlock>>>(reflectance, bioscat.reflectance, i, j, n_obs);
                cudaDeviceSynchronize();
            }
            else {
                for (int k = 0; k < n_obs; k++) {
                    reflectance.setHostValue(k,j,i,bioscat.reflectance.getHostValue(k));
                }   
            }
            
            
        }
    }
    showProgressBar(1.0);
}

void computeReflectanceMatrix(Nanostructure proposedNanostructure, BioScat bioscat, RealMatrix reflectance, RealMatrix lambdas, RealMatrix betas, int num_lambdas, int num_betas, int n_obs, bool deviceComputation)  {

    proposedNanostructure.f.toDevice();
    bioscat.getSegments(proposedNanostructure);

    for (int i = 0; i < num_lambdas; i++) {
        double lambda = lambdas.getHostValue(i);
        bioscat.reset(); // Set fields to zero
        bioscat.prepareForward(lambda);
         double start_test = omp_get_wtime();
        for (int polarisation = 1; polarisation <= 2; polarisation++) {
            bioscat.forwardSolver(polarisation);
            bioscat.computeScatteredSubFields();
            //bioscat.computeReflectedSubFields();
            bioscat.computeIncidentSubFields();
        }

       
        for (int j = 0; j < num_betas; j++) {
            double beta = betas.getHostValue(j);
            
            bioscat.computeScatteredFields(beta);
            //bioscat.computeReflectedFields(beta);
            bioscat.computeIncidentFields(beta);
            bioscat.computeReflectance();
            
            if (deviceComputation) {
                // Blocks and threads
                dim3 dimBlock(256);
                dim3 dimGrid((n_obs + dimBlock.x - 1)/dimBlock.x);
                storeReflectanceKernel<<<dimGrid, dimBlock>>>(reflectance, bioscat.reflectance, i, j, n_obs);
                cudaDeviceSynchronize();
            }
            else {
                for (int k = 0; k < n_obs; k++) {
                    reflectance.setHostValue(k,j,i,bioscat.reflectance.getHostValue(k));
                }   
            }
            
            
        }
    }
}

}