#include "../lib/generateArtificialData.h"

void executeGenerateArtificialData(double *x, double*y, int n ,char* protein_structure,  int num_segments,int total_grid_points,double * betas, double * lambdas, int num_betas, int num_lambdas) {

    generateArtificialData(x, y, n, protein_structure, num_segments, total_grid_points, betas, lambdas, num_betas, num_lambdas);

}