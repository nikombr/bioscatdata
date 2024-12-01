#include "../lib/farFieldPattern.h"

void executeFarFieldPattern(double * phi, int n, char* protein_structure, int num_segments, int total_grid_points, double beta, double lambda, int deviceComputation_int) {

    farFieldPattern(phi, n, protein_structure, num_segments,total_grid_points, beta, lambda, deviceComputation_int);

}