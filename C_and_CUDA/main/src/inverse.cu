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
#include "../lib/computeReflectanceMatrix.h"
extern "C" {

using namespace std;
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

int countNumbersInFile(char* filename) {
    double num;
    FILE * file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n",filename);
        return 0;
    }

    int count = 0;

    while (fscanf(file, "%e", &num) == 1) {
        count++;
    }

    // Close the file
    fclose(file);

    return count;
}

void getNumPoints(int * n_obs, int * num_lambdas, int * num_betas, char * protein_structure, int total_grid_points) {
    // Get number of data points
    char filename[256];

    sprintf(filename, "../../../Data/artificial_data/%s/num_segments_1_total_grid_points_%d/x_obs.txt",protein_structure,total_grid_points);

    * n_obs = countNumbersInFile(filename);

    sprintf(filename, "../../../Data/artificial_data/%s/num_segments_1_total_grid_points_%d/lambdas.txt",protein_structure,total_grid_points);

    * num_lambdas = countNumbersInFile(filename);
   
    sprintf(filename, "../../../Data/artificial_data/%s/num_segments_1_total_grid_points_%d/betas.txt",protein_structure,total_grid_points);

    * num_betas = countNumbersInFile(filename);
    
}

void loadData(RealMatrix trueReflectance, RealMatrix lambdas, RealMatrix betas, RealMatrix x_obs, RealMatrix y_obs, char * protein_structure, int total_grid_points) {
    char filename[256];
    sprintf(filename, "../../../Data/artificial_data/%s/num_segments_1_total_grid_points_%d/reflectance.txt",protein_structure,total_grid_points);
    trueReflectance.loadVector(filename);
    sprintf(filename, "../../../Data/artificial_data/%s/num_segments_1_total_grid_points_%d/lambdas.txt",protein_structure,total_grid_points);
    lambdas.loadVector(filename);
    sprintf(filename, "../../../Data/artificial_data/%s/num_segments_1_total_grid_points_%d/betas.txt",protein_structure,total_grid_points);
    betas.loadVector(filename);
    sprintf(filename, "../../../Data/artificial_data/%s/num_segments_1_total_grid_points_%d/x_obs.txt",protein_structure,total_grid_points);
    x_obs.loadVector(filename);
    sprintf(filename, "../../../Data/artificial_data/%s/num_segments_1_total_grid_points_%d/y_obs.txt",protein_structure,total_grid_points);
    y_obs.loadVector(filename);
}

void initCoordInNanostructure(Nanostructure nanostructure, char * protein_structure, int total_grid_points) {
    double val;
    char filename[256];
    sprintf(filename, "../../../Data/nanostructures/2D/%s_x_%d.txt", protein_structure, total_grid_points);
    //printf("filename = %s\n",filename);

    FILE *file;
    file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        printf("File: %s\n",filename);
        return;
    }
    for (int i = 0; i < total_grid_points; i++) {
        fscanf(file, "%lf,", &val);  // Reading each value into the array
        nanostructure.x.setHostValue(i, val);
    }
    fclose(file);

    nanostructure.x.toDevice();
    cudaDeviceSynchronize();
}

double computeLogLikelihood(RealMatrix trueReflectance, RealMatrix reflectance, bool deviceComputation) {

    if (false) {
        return 0;
    }
    else {
        reflectance.toHost();
        double beta = 1e2;
        double L = 0;
        //#pragma omp parallel for reduction(+:L)
        for (int i = 0; i < reflectance.rows; i++) {
            for (int j = 0; j < reflectance.cols; j++) {
                for (int k = 0; k < reflectance.depth; k++) {
                    double val = abs(trueReflectance.getHostValue(i,j,k) - reflectance.getHostValue(i,j,k));
                    L += val*val;
                }
            }
        }
        //int N = reflectance.rows * reflectance.cols * reflectance.depth;
        return -0.5*beta*sqrt(L);
    }
    

}

void swapPointers(RealMatrix x, RealMatrix y) {
    double * temp = x.getHostPointer();
    x.setHostPointer(y.getHostPointer());
    y.setHostPointer(temp);
}


double computeInverseStep(Nanostructure proposedNanostructure, BioScat bioscat, RealMatrix reflectance, RealMatrix trueReflectance, RealMatrix lambdas, RealMatrix betas, int num_lambdas, int num_betas, int n_obs, bool deviceComputation) {
    computeReflectanceMatrix(proposedNanostructure, bioscat, reflectance, lambdas, betas, num_lambdas, num_betas, n_obs, deviceComputation);
    return computeLogLikelihood(trueReflectance, reflectance,deviceComputation);
}

void inverse(char * protein_structure, int num_segments, int total_grid_points, double * hyper, int num, int type_covfunc, double delta, int maxiter, char * filename) {

    double start, stop; // Time measurement
    double Lprev, Lstar, alpha, u, logPrior;
    double shift = 3e-8;
    double scale = 0.75*1e-8;
    double * temp_h, * temp_d;
    bool deviceComputation = true;
    
    // Files for output
    FILE *file, *logfile, *logfile_accepted;
    char current_file_name[256];
    sprintf(current_file_name,"../../../Results/inverse/%s_output.txt",filename);
    file = fopen(current_file_name, "w");
    if (file == NULL) {
        perror("Error opening output file");
        return;
    }
    sprintf(current_file_name,"../../../Results/inverse/%s_log.txt",filename);
    logfile = fopen(current_file_name, "w");
    if (file == NULL) {
        perror("Error opening output file");
        return;
    }
    sprintf(current_file_name,"../../../Results/inverse/%s_log_accepted.txt",filename);
    logfile_accepted = fopen(current_file_name, "w");
    if (file == NULL) {
        perror("Error opening output file");
        return;
    }

    // Get number of data points
    int n_obs, num_lambdas, num_betas;
    getNumPoints(&n_obs, &num_lambdas, &num_betas, protein_structure, total_grid_points);
    printf("(n_obs, n_lambdas, n_beta) = (%d, %d, %d)\n",n_obs,num_lambdas,num_betas);

    // Allocate matrices for data
    RealMatrix trueReflectance = RealMatrix(n_obs, num_betas, num_lambdas);
    RealMatrix lambdas         = RealMatrix(num_lambdas);
    RealMatrix betas           = RealMatrix(num_betas);
    RealMatrix x_obs           = RealMatrix(n_obs);
    RealMatrix y_obs           = RealMatrix(n_obs);
    RealMatrix f               = RealMatrix(total_grid_points);
    RealMatrix fstar           = RealMatrix(total_grid_points);
    RealMatrix phi             = RealMatrix(total_grid_points);
    RealMatrix reflectance     = RealMatrix(n_obs, num_betas, num_lambdas);

    // Load data into matrices
    loadData(trueReflectance, lambdas, betas, x_obs, y_obs, protein_structure, total_grid_points);
   
    // Initialize some of the proposed nanostructures
    Nanostructure proposedNanostructure = Nanostructure(total_grid_points);
    initCoordInNanostructure(proposedNanostructure, protein_structure, total_grid_points);
    
    // Setup Gaussian Process for realisations
    start = omp_get_wtime();
    GaussianProcess GP = GaussianProcess(total_grid_points, hyper, num, type_covfunc);
    stop = omp_get_wtime();

    printf("Initialization and allocation: %.4f seconds\n\n", stop - start);

    start = omp_get_wtime();
    GP.covariance_matrix();
    stop = omp_get_wtime();

    printf("Computing covariance matrix: %.4f seconds\n\n", stop - start);

    start = omp_get_wtime();
    GP.compute_inverse();
    stop = omp_get_wtime();

    printf("Computing inverse of covariance matrix: %.4f seconds\n\n", stop - start);

    start = omp_get_wtime();
    GP.cholesky();
    stop = omp_get_wtime();

    printf("Cholesky factorization: %.4f seconds\n\n", stop - start);

    
    // Seed the random number generator with the current time
    srand(time(NULL));
    double minimum = 0;
    while (minimum < 1e-8) {
        /*GP.realisation();

        for (int i = 0; i < total_grid_points; i++) {
            f.setHostValue(i,GP.p_h[i]);
        }*/

        for (int i = 0; i < total_grid_points; i++) {
            f.setHostValue(i,0.0);
        }

        for (int i = 0; i < total_grid_points; i++) {
            proposedNanostructure.f.setHostValue(i,f.getHostValue(i)*scale + shift);
        }

        minimum = proposedNanostructure.f.findMin();
        printf("minimum = %e\n",minimum);
    }
    
    BioScat bioscat = BioScat(protein_structure, num_segments, total_grid_points, deviceComputation);

    bioscat.setupObservationPoints(x_obs.getHostPointer(), y_obs.getHostPointer(), n_obs);
    bioscat.allocateSegments();
    int acc = 0; // Count how many curves are accepted

    f.toDevice();
    logPrior = GP.compute_prior(f.getDevicePointer());

    Lprev = computeInverseStep(proposedNanostructure, bioscat, reflectance, trueReflectance, lambdas, betas, num_lambdas, num_betas, n_obs, deviceComputation);
    printf("hej\n");
    for (int j = 0; j < f.rows; j++) {
        fprintf(file, "%e ", proposedNanostructure.f.getHostValue(j));
    }
    fprintf(file, "\n");
    fflush(file);
    /*Lprev = computeLogLikelihood(trueReflectance, reflectance);
    printf("Lprev = %e\n", Lprev);
    computeReflectanceMatrix(proposedNanostructure, bioscat, reflectance, lambdas, betas, num_lambdas, num_betas, n_obs);
    Lprev = computeLogLikelihood(trueReflectance, reflectance);
    printf("Lprev = %e\n", Lprev);*/

    
    fprintf(logfile, "accepted\tLprev\tLstar\talpha\tminimum\ttime\n");
    fprintf(logfile_accepted, "%e %e %e %e %e %e\n",Lprev, exp(Lprev),logPrior, exp(logPrior), Lprev + logPrior, exp(Lprev + logPrior));

    int status;
    for (int n = 0; n < maxiter; n++) {
        
        
        GP.realisation();

        for (int i = 0; i < total_grid_points; i++) {
            phi.setHostValue(i,GP.p_h[i]);
        }
        

        for (int i = 0; i < total_grid_points; i++) {
            fstar.setHostValue(i,sqrt(1 - 2*delta)*f.getHostValue(i) + sqrt(2*delta)*phi.getHostValue(i));
        }
        for (int i = 0; i < total_grid_points; i++) {
            proposedNanostructure.f.setHostValue(i,fstar.getHostValue(i)*scale + shift);
        }
        minimum = proposedNanostructure.f.findMin();
        //printf("minimum = %e\n",minimum);
        double start = omp_get_wtime();
        Lstar = computeInverseStep(proposedNanostructure, bioscat, reflectance, trueReflectance, lambdas, betas, num_lambdas, num_betas, n_obs, deviceComputation);
        double end = omp_get_wtime();
  
        /*if (status == EXIT_SUCCESS) {
        // Compute log-likelihood
        Lstar = computeLogLikelihood(trueReflectance, reflectance);
        if (minimum < 1e-8) Lstar *= 100;
        //printf("(Lstar, Lprev) = (%f, %f)\n", Lstar, Lprev);*/

        // Compute probability of accepting curve
        
        alpha = std::min((double)1,exp(Lstar-Lprev));
        if (minimum < 1e-8) alpha = 0.0;
        //printf("alpha = %f\n", alpha);

        // Generate random number
        u = ((double) rand())/((double) RAND_MAX);
        //printf("u = %f\n",u);

        if (u < alpha) {
            acc++;
            //printf("ACCEPTED %d\n",acc++);
            //swapPointers(f, fstar);
            temp_h = f.getHostPointer();
            temp_d = f.getDevicePointer();
            f.setHostPointer(fstar.getHostPointer());
            f.setDevicePointer(fstar.getDevicePointer());
            fstar.setHostPointer(temp_h);
            fstar.setDevicePointer(temp_d);
            Lprev = Lstar;
            for (int j = 0; j < f.rows; j++) {
                fprintf(file, "%e ", proposedNanostructure.f.getHostValue(j));
            }
            fprintf(file, "\n");
            fflush(file);
            f.toDevice();
            logPrior = GP.compute_prior(f.getDevicePointer());
            fprintf(logfile_accepted, "%e %e %e %e %e %e\n",Lprev, exp(Lprev),logPrior, exp(logPrior), Lprev + logPrior, exp(Lprev + logPrior));
            fflush(logfile_accepted);
        }

        fprintf(logfile, "%d\t\t%f\t%f\t%f\t%e\t%e\n", acc, Lprev, Lstar, alpha, minimum, end - start);
        fflush(logfile);
        /*}
        else {
            printf("Skipped a curve!\n");
        }*/
          
    }

    fclose(file);
    fclose(logfile);

    trueReflectance.free();
    lambdas.free();
    betas.free();
    x_obs.free();
    y_obs.free();
    GP.free();
    //bioscat.free();




}


}