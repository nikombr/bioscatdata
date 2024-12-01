
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include <omp.h>
//#include <cblas.h>
#include <math.h>
#include "../lib/BioScat.h"
#include "../lib/Segment.h"
#include "../lib/RealMatrix.h"
#include "../lib/combinePolarisation.h"
#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
extern "C" {
using namespace std;

BioScat::BioScat(char* protein_structure, int num_segments, int total_grid_points) {

    this->protein_structure = protein_structure;
    this->num_segments = num_segments;
    this->total_grid_points = total_grid_points;

}

BioScat::BioScat(char* protein_structure, int num_segments, int total_grid_points, bool deviceComputation) {

    this->protein_structure = protein_structure;
    this->num_segments = num_segments;
    this->total_grid_points = total_grid_points;
    this->deviceComputation = deviceComputation;

}

void BioScat::free() {

    for (int i = 0; i < num_segments; i++) {
        segments[i].free();
    }

    delete[] segments;
    reflectance.free();
    x_obs.free();
    y_obs.free();

    E_scat.free();
    H_scat.free();
    E_inc.free();
    H_inc.free();
    E_ref.free();
    H_ref.free();
    for (int i = 0; i < 2; i++) {
        E_scat_pol[i].free();
        H_scat_pol[i].free();
        E_inc_pol[i].free();
        H_inc_pol[i].free();
        E_ref_pol[i].free();
        H_ref_pol[i].free();
    }
    //cudaDeviceReset();
    if (printOutput) printf("DESTRUCTED!\n");

}

void BioScat::getSegments() {

    if (deviceComputation) {
        if (printOutput) printf("We are computing on the device!\n");
    }
    else {
        if (printOutput) printf("We are computing on the host!\n");
    }

    double start = omp_get_wtime();

    for (int i = 0; i < num_segments; i++) {
        segments[i].current_segment = i;
        segments[i].n_obs = n_obs;
        segments[i].setup(this->nanostructure, total_grid_points, num_segments);
    }


    double end = omp_get_wtime();
    if (printOutput) printf("\nIt took %e seconds to setup the segments!\n\n",end-start);

}

void BioScat::allocateSegments() {
    int sideSteps      = 0.75*total_grid_points;
    int segment_length = total_grid_points / num_segments;
    int n_top          = segment_length - 2;
    int n_bottom       = segment_length - 2;
    int n_right        = sideSteps - 1;
    int n_left         = sideSteps - 1;
    int n_int          = n_top + n_right + n_bottom + n_left - 16;
    int n_ext          = 2*(segment_length - 1) + sideSteps + sideSteps;
    int n_test         = n_top + n_bottom + n_right + n_left;
    int n = std::max(n_obs, n_test);
    if (printOutput) printf("n_test:  \t%d\nn_top:  \t%d\nn_right:  \t%d\nn_bottom:\t%d\nn_left:  \t%d\nn_int:  \t%d\nn_ext:  \t%d\nn_obs:  \t%d\nn:      \t%d\n", n_test, n_top, n_right, n_bottom, n_left, n_int, n_ext, n_obs, n);
    this->segments = new Segment[num_segments];
    for (int i = 0; i < num_segments; i++) {
        segments[i].n_obs    = n_obs;
        segments[i].n_int    = n_int;
        segments[i].n_ext    = n_ext;
        segments[i].n_test   = n_test;
        segments[i].n_top    = n_top;
        segments[i].n_right  = n_right;
        segments[i].n_bottom = n_bottom;
        segments[i].n_left   = n_left;
        segments[i].segment_length = segment_length;
        segments[i].deviceComputation = deviceComputation;
        segments[i].allocate();
        
    }
}

void BioScat::getSegments(Nanostructure nanostructure) {

    double start = omp_get_wtime();

    for (int i = 0; i < num_segments; i++) {
        //segments[i] = Segment();
        segments[i].current_segment = i;
        segments[i].setup(nanostructure, total_grid_points, num_segments);
    }

    double end = omp_get_wtime();
    if (printOutput) printf("\nIt took %e seconds to setup the segments!\n\n",end-start);
}

void BioScat::prepareForward(double lambda) {
    for (int i = 0; i < num_segments; i++) {
        segments[i].newWavelength(lambda);
    }
}

void BioScat::prepareForward(double beta, double lambda) {
    this->beta = beta;
    for (int i = 0; i < num_segments; i++) {
        segments[i].newWavelength(lambda);
    }
}

void BioScat::forwardSolver(int polarisation) {

    this->polarisation = polarisation;

    double start, end, start_inner, end_inner;
    start = omp_get_wtime();
    #pragma omp parallel for num_threads(num_segments)
    for (int i = 0; i < num_segments; i++) {
        segments[i].polarisation = polarisation;

        start_inner = omp_get_wtime();
        
        segments[i].computeFieldsForLinearSystem();
        segments[i].setupRightHandSide();
        segments[i].setupSystemMatrix();
        //segments[i].freeScatteredFields();
        //segments[i].freeInteriorFields();
        //segments[i].freeIncidentFields();
        //segments[i].freeReflectedFields();
        
        segments[i].solveLinearSystem();
      
        end_inner = omp_get_wtime();
        if (printOutput) printf("\nIt took %.4e seconds to solve the linear system for segment %d.\n\n",end_inner - start_inner, i + 1);
        
    }
    end = omp_get_wtime();
    if (printOutput) printf("\nIt took %.4e seconds to solve all the linear systems.\n\n",end - start);
}

void BioScat::setupObservationPoints(double *x, double*y, int n) {
    n_obs = n;
    x_obs = RealMatrix(n);
    y_obs = RealMatrix(n);
    for (int i = 0; i < n; i++) x_obs.setHostValue(i, x[i]);
    for (int i = 0; i < n; i++) y_obs.setHostValue(i, y[i]);
    x_obs.toDevice();
    y_obs.toDevice();

    // Allocate fields
    E_scat = Field(n);
    H_scat = Field(n);
    E_inc  = Field(n);
    H_inc  = Field(n);
    for (int i = 0; i < 2; i++) {
        E_scat_pol[i] = Field(n);
        H_scat_pol[i] = Field(n);
        E_inc_pol[i]  = Field(n);
        H_inc_pol[i]  = Field(n);
    }

    // Allocate reflectance array
    reflectance = RealMatrix(n);
}

void BioScat::setupObservationPoints(double *phi, int n) {
    n_obs = n;
    phi_obs = RealMatrix(n);
    for (int i = 0; i < n; i++) phi_obs.setHostValue(i, phi[i]);
    phi_obs.toDevice();

    // Allocate fields
    F = RealMatrix(n);
    for (int i = 0; i < 2; i++) {
        F_pol[i] = ComplexMatrix(n);
    }

    // Allocate reflectance array
    reflectance = RealMatrix(n);
}

void BioScat::computeScatteredFields() {

    computeScatteredFields(beta);
}


void BioScat::computeScatteredFields(double beta) {
    combinePolarisation(E_scat_pol, E_scat, beta, deviceComputation);
    combinePolarisation(H_scat_pol, H_scat, beta, deviceComputation);
    
}

void BioScat::computeIncidentFields() {
    
    computeIncidentFields(beta);
}


void BioScat::computeIncidentFields(double beta) {

    combinePolarisation(E_inc_pol, E_inc, beta, deviceComputation);
    combinePolarisation(H_inc_pol, H_inc, beta, deviceComputation);
}

void BioScat::computeReflectedFields() {
    computeReflectedFields(beta);
}


void BioScat::computeReflectedFields(double beta) {

    combinePolarisation(E_ref_pol, E_ref, beta, deviceComputation);
    combinePolarisation(H_ref_pol, H_ref, beta, deviceComputation);

    
}

void BioScat::dumpFields() {

    if (deviceComputation) {
        E_scat.toHost();
        E_ref.toHost();
        E_inc.toHost();
        H_scat.toHost();
        H_ref.toHost();
        H_inc.toHost();
    }
    

    char filename[256];

    // Save scattered electric fields
    sprintf(filename, "../../../Results/forward/Ex_scat.txt");
    E_scat.x.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Ey_scat.txt");
    E_scat.y.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Ez_scat.txt");
    E_scat.z.dumpResult(filename);

    // Save scattered magnetic fields
    sprintf(filename, "../../../Results/forward/Hx_scat.txt");
    H_scat.x.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Hy_scat.txt");
    H_scat.y.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Hz_scat.txt");
    H_scat.z.dumpResult(filename);

    // Save incident electric fields
    sprintf(filename,"../../../Results/forward/Ex_inc.txt");
    E_inc.x.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Ey_inc.txt");
    E_inc.y.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Ez_inc.txt");
    E_inc.z.dumpResult(filename);

    // Save incident magnetic fields
    sprintf(filename,"../../../Results/forward/Hx_inc.txt");
    H_inc.x.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Hy_inc.txt");
    H_inc.y.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Hz_inc.txt");
    H_inc.z.dumpResult(filename);

     // Save reflected electric fields
    /*sprintf(filename,"../../../Results/forward/Ex_ref.txt");
    E_ref.x.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Ey_ref.txt");
    E_ref.y.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Ez_ref.txt");
    E_ref.z.dumpResult(filename);

    // Save reflected magnetic fields
    sprintf(filename,"../../../Results/forward/Hx_ref.txt");
    H_ref.x.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Hy_ref.txt");
    H_ref.y.dumpResult(filename);
    sprintf(filename,"../../../Results/forward/Hz_ref.txt");
    H_ref.z.dumpResult(filename);*/
}

void BioScat::dumpFarFields() {

    if (deviceComputation) {
        F.toHost();
    }

    char filename[256];

    // Save scattered electric fields
    sprintf(filename, "../../../Results/forward/farFieldPattern.txt");
    F.dumpResult(filename);
}

void BioScat::reset() {
    if (deviceComputation) {
        // Initialize all arrays to zero on the host
        for (int i = 0; i < 2; i++) {
            E_scat_pol[i].setDeviceZero();
            H_scat_pol[i].setDeviceZero();
            E_inc_pol[i].setDeviceZero();
            H_inc_pol[i].setDeviceZero();
            E_ref_pol[i].setDeviceZero();
            H_ref_pol[i].setDeviceZero();
        }
    }
    else {
        // Initialize all arrays to zero on the host
        for (int i = 0; i < 2; i++) {
            E_scat_pol[i].setHostZero();
            H_scat_pol[i].setHostZero();
            E_inc_pol[i].setHostZero();
            H_inc_pol[i].setHostZero();
            E_ref_pol[i].setHostZero();
            H_ref_pol[i].setHostZero();
        }
    }
}

}