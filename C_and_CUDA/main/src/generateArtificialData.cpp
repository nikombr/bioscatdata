#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include "../lib/ComplexMatrix.h"
#include "../lib/BioScat.h"
#include "../lib/computeReflectanceMatrix.h"
#include "../lib/RealMatrix.h"
extern "C" {
using namespace std;

void generateArtificialData(double *x, double*y, int n, char * protein_structure, int num_segments, int total_grid_points, double * betas, double * lambdas, int num_betas, int num_lambdas) {
    
    RealMatrix reflectance = RealMatrix(n, num_betas, num_lambdas);
   
    bool deviceComputation = true;
    //double start = omp_get_wtime();
    BioScat bioscat = BioScat(protein_structure, num_segments, total_grid_points, deviceComputation);
    bioscat.printOutput = false;
    
    bioscat.setupObservationPoints(x, y, n);

    bioscat.allocateSegments();
    double start = omp_get_wtime();
    bioscat.getNanostructure();

    computeReflectanceMatrixGetData(bioscat, reflectance, lambdas, betas, num_lambdas, num_betas, n, deviceComputation);

    /*bioscat.getSegments();

    double scat_time = 0;
    double ref_time = 0;
    double forward_time = 0;
    double start2, stop2;
    
    for (int i = 0; i < num_lambdas; i++) {
        bioscat.reset();
        double lambda = lambdas[i];
        bioscat.prepareForward(lambda);

        for (int polarisation = 1; polarisation <= 2; polarisation++) {
            start2 = omp_get_wtime();
            bioscat.forwardSolver(polarisation);
            stop2 = omp_get_wtime();
            forward_time += stop2 - start2;
            bioscat.computeScatteredSubFields();
            bioscat.computeReflectedSubFields();
            bioscat.computeIncidentSubFields();
        }
        
        for (int j = 0; j < num_betas; j++) {
            showProgressBar((i*num_betas + j) / ((double) num_lambdas*num_betas));
            double beta = betas[j];
            start2 = omp_get_wtime();
            bioscat.computeScatteredFields(beta);
            stop2 = omp_get_wtime();
            scat_time += stop2 - start2;
            start2 = omp_get_wtime();
            bioscat.computeReflectedFields(beta);
            stop2 = omp_get_wtime();
            ref_time += stop2 - start2;

            bioscat.computeIncidentFields(beta);
            //double start = omp_get_wtime();
            bioscat.computeReflectance();
            //double end = omp_get_wtime();
            //printf("the time was %e\n",end-start);
            
            for (int k = 0; k < n; k++) {
                reflectance.setHostValue(k,i,j,bioscat.reflectance.getHostValue(k));
            }
            
        }
    }
    showProgressBar(1.0);*/
    double end = omp_get_wtime();
    printf("\nIt took %.4f to compute the reflectance!\n",end-start);

    /*printf("\nScattering time = %f\n\n",scat_time);
    printf("\nReflected time = %f\n\n",ref_time);
    printf("\nForward time = %f\n\n",forward_time);*/

    char filename[256];
    sprintf(filename, "../../../Data/artificial_data/temp/reflectance.txt");
    reflectance.toHost();
    reflectance.dumpVector(filename);

    bioscat.free();


}



}
