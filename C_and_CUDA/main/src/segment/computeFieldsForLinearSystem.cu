#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>
#include "../../lib/Segment.h"
extern "C" {
#include "../../lib/RealMatrix.h"
using namespace std;

void Segment::computeFieldsForLinearSystem() {

    
    computeIncidentFieldVectors(y_test);
    //computeReflectedFieldVectors(y_test);
    computeScatteredFieldMatrices(x_test, y_test, false);
    computeInteriorFieldMatrices(x_test, y_test);

}

}