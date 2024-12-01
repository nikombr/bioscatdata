#ifndef _BIOSCAT_H
#define _BIOSCAT_H
#include "Segment.h"
extern "C" {
#include "Nanostructure.h"
#include "Field.h"
#include "RealMatrix.h"
#include "GP/GaussianProcess.h"

class BioScat {

    private:
        char * protein_structure;    // Type of protein structure. Either "Retinin2x2" or "demoleus2x2"
        Nanostructure nanostructure; // Information stored about the specific nanostructure 
        int total_grid_points = 1000;       // The number of grid points used along each axis
        Segment * segments;
        int num_segments;
        Field E_scat;
        Field H_scat;
        Field E_inc;
        Field H_inc;
        Field E_ref;
        Field H_ref;
        RealMatrix F;
        Field E_scat_pol[2];
        Field H_scat_pol[2];
        Field E_inc_pol[2];
        Field H_inc_pol[2];
        Field E_ref_pol[2];
        Field H_ref_pol[2];
        ComplexMatrix F_pol[2];
        bool deviceComputation = false; // True if we should compute on the device
        int polarisation; // Polarisation polarisation, 1 or 2
        double beta;
        

        

    public:
        RealMatrix reflectance;
        GaussianProcess GP;
        RealMatrix x_obs;
        RealMatrix y_obs;
        RealMatrix phi_obs;
        int n_obs;
        bool printOutput = false;
        int status;
        BioScat() {

        }
        BioScat(char* protein_structure, int num_segments, int total_grid_points);
        BioScat(char* protein_structure, int num_segments, int total_grid_points, bool deviceComputation);
        void free();
        void getNanostructure();                                        // Set up nanostructure from protein_structure
        void getSegments();
        void getSegments(Nanostructure nanostructure);
        void forwardSolver(int polarisation);
        void setupObservationPoints(double *x, double*y, int n);
        void computeScatteredFields();
        void computeScatteredFields(double lambda);
        void computeScatteredSubFields();
        void computeReflectedFields();
        void computeReflectedSubFields();
        void computeReflectedFields(double lambda);
        void computeIncidentFields();
        void computeIncidentSubFields();
        void computeIncidentFields(double lambda);
        void dumpFields();
        void dumpFarFields();
        void computeReflectance();
        void prepareForward(double beta, double lambda);
        void prepareForward(double lambda);
        void inverseSolver();
        void setupGaussianProcess();
        void preConditionedCrankNicholson();
        void reset(); // Sets pols to zero
        void allocateSegments();
        void setupObservationPoints(double *phi, int n);
        void computeFarFieldPattern();
        void combineFarFieldPattern();

        



};
}
#endif

/*
    private:        
        int n;              // number of points for estimating plane
        double *hyper_h;    // hyperparameters on host
        double *hyper_d;    // hyperparameters on device
        int num;            // number of hyperparameters
        int dim;            // dimension of the problem, either 1 for curve or 2 for plane
        double *x_h;        // x coordinates for estimating plane on host
        double *x_d;        // x coordinates for estimating plane on device
        double *y_h;        // y coordinates for estimating plane on host
        double *y_d;        // y coordinates for estimating plane on device
        double **M_d;       // covariance matrix and later lower triangular matrix from Cholesky factorization on device
        int type_covfunc;
    public:
        double **M_h;       // covariance matrix and later lower triangular matrix from Cholesky factorization on host
        double *M_log;      // M_d[0] on device
        bool device;
        double *p_h;        // random vector and later height of plane in location (x,y) on host
        double *p_d;        // random vector and later height of plane in location (x,y) on device
        GaussianProcess(double* x_h, double* y_h, int n, double* hyper, int num, int dim, int dev, int type_covfunc);  // Constructer, sets default values and allocates
        ~GaussianProcess();                                                                 // Destructer
        void covariance_matrix();                                                           // Computes covariance matrix K
        void cholesky();                                                                    // Does cholesky factorization of K to compute L
        void generate_random_vector();                                                      // Generates random vector p
        void realisation();                                                                 // Computes realisation of the Gaussian process from L
};

#endif*/