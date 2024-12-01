# Makefile 
#SOURCES      	= src/BioScat.cpp src/forward.cpp
#OBJECTS      	= obj/BioScat.o   obj/forward.o
#SOURCES_SEG     = src/Segment.cu
#OBJECTS_SEG     = obj/Segment.o
#SOURCES_MATRIX 	= src/ComplexMatrix.cu src/RealMatrix.cu
#OBJECTS_MATRIX 	= obj/ComplexMatrix.o  obj/RealMatrix.o
#SOURCES_GP 		= src/GP/cholesky.cu src/GP/cudaMalloc2d.cu src/GP/covariance_matrix.cu  src/GP/realisation.cu src/GP/gaussian_process_inner.cpp src/GP/GaussianProcess.cpp
#OBJECTS_GP 		= obj/GP/cholesky.o  obj/GP/cudaMalloc2d.o  obj/GP/covariance_matrix.o   obj/GP/realisation.o  obj/GP/gaussian_process_inner.o   obj/GP/GaussianProcess.o
#SOURCES_2D 		= src/2D/getNanostructure.cpp src/2D/Segment.cu
#OBJECTS_2D 		= obj/2D/getNanostructure.o   obj/2D/Segment.o

# Files and dependencies

SOURCE_REALMATRIX = src/RealMatrix.cu
OBJECT_REALMATRIX = obj/RealMatrix.o
DEPEND_REALMATRIX = 

SOURCE_COMPLEXMATRIX = src/ComplexMatrix.cu
OBJECT_COMPLEXMATRIX = obj/ComplexMatrix.o
DEPEND_COMPLEXMATRIX = 

SOURCE_GET_NANOSTRUCTURE = src/getNanostructure.cpp
OBJECT_GET_NANOSTRUCTURE = obj/getNanostructure.o
DEPEND_GET_NANOSTRUCTURE = $(OBJECT_REALMATRIX)

SOURCE_SEGMENT_COMPUTE_FIELD_MATRICES = src/segment/computeFieldMatrices.cu
OBJECT_SEGMENT_COMPUTE_FIELD_MATRICES = obj/segment/computeFieldMatrices.o 
DEPEND_SEGMENT_COMPUTE_FIELD_MATRICES = $(OBJECT_REALMATRIX)

SOURCE_SEGMENT_COMPUTE_FAR_FIELD_PATTERN = src/segment/computeFarFieldPattern.cu
OBJECT_SEGMENT_COMPUTE_FAR_FIELD_PATTERN = obj/segment/computeFarFieldPattern.o 
DEPEND_SEGMENT_COMPUTE_FAR_FIELD_PATTERN = $(OBJECT_REALMATRIX)

SOURCE_COMPUTE_FAR_FIELD_PATTERN = src/computeFarFieldPattern.cu
OBJECT_COMPUTE_FAR_FIELD_PATTERN = obj/computeFarFieldPattern.o 
DEPEND_COMPUTE_FAR_FIELD_PATTERN = $(OBJECT_REALMATRIX) $(OBJECT_SEGMENT_COMPUTE_FAR_FIELD_PATTERN)

SOURCE_SEGMENT_COMPUTE_SCATTERED_SUB_FIELDS = src/segment/computeScatteredSubFields.cu
OBJECT_SEGMENT_COMPUTE_SCATTERED_SUB_FIELDS = obj/segment/computeScatteredSubFields.o 
DEPEND_SEGMENT_COMPUTE_SCATTERED_SUB_FIELDS = $(OBJECT_REALMATRIX)

SOURCE_SEGMENT_COMPUTE_FIELD_VECTORS = src/segment/computeFieldVectors.cu
OBJECT_SEGMENT_COMPUTE_FIELD_VECTORS = obj/segment/computeFieldVectors.o
DEPEND_SEGMENT_COMPUTE_FIELD_VECTORS = $(OBJECT_REALMATRIX)

SOURCE_SEGMENT_COMPUTE_FIELDS_FOR_LINEAR_SYSTEM = src/segment/computeFieldsForLinearSystem.cu
OBJECT_SEGMENT_COMPUTE_FIELDS_FOR_LINEAR_SYSTEM = obj/segment/computeFieldsForLinearSystem.o
DEPEND_SEGMENT_COMPUTE_FIELDS_FOR_LINEAR_SYSTEM = $(OBJECT_REALMATRIX) $(OBJECT_SEGMENT_COMPUTE_FIELD_MATRICES) $(OBJECT_SEGMENT_COMPUTE_FIELD_VECTORS)

SOURCE_SEGMENT_SEGMENT = src/segment/Segment.cpp
OBJECT_SEGMENT_SEGMENT = obj/segment/Segment.o
DEPEND_SEGMENT_SEGMENT = $(OBJECT_REALMATRIX)

SOURCE_SEGMENT_KERNELS = src/segment/setup/kernels.cu
OBJECT_SEGMENT_KERNELS = obj/segment/setup/kernels.o
DEPEND_SEGMENT_KERNELS = $(OBJECT_REALMATRIX) 

SOURCE_SEGMENT_SETUP_1 = src/segment/setup/computeExteriorPointsAndNormalVectors.cu
OBJECT_SEGMENT_SETUP_1 = obj/segment/setup/computeExteriorPointsAndNormalVectors.o
DEPEND_SEGMENT_SETUP_1 = $(OBJECT_REALMATRIX) $(OBJECT_SEGMENT_KERNELS)

SOURCE_SEGMENT_SETUP_2 = src/segment/setup/computeTestPoints.cu
OBJECT_SEGMENT_SETUP_2 = obj/segment/setup/computeTestPoints.o
DEPEND_SEGMENT_SETUP_2 = $(OBJECT_REALMATRIX) 

SOURCE_SEGMENT_SETUP_3 = src/segment/setup/computeInteriorPoints.cu
OBJECT_SEGMENT_SETUP_3 = obj/segment/setup/computeInteriorPoints.o
DEPEND_SEGMENT_SETUP_3 = $(OBJECT_REALMATRIX) $(OBJECT_SEGMENT_KERNELS)

SOURCE_SEGMENT_SETUP = src/segment/setup.cpp
OBJECT_SEGMENT_SETUP = obj/segment/setup.o
DEPEND_SEGMENT_SETUP = $(OBJECT_REALMATRIX) $(OBJECT_SEGMENT_SETUP_1) $(OBJECT_SEGMENT_SETUP_2) $(OBJECT_SEGMENT_SETUP_3)

SOURCE_SEGMENT_SETUP_RIGHT_HAND_SIDE = src/segment/setupRightHandSide.cu
OBJECT_SEGMENT_SETUP_RIGHT_HAND_SIDE = obj/segment/setupRightHandSide.o
DEPEND_SEGMENT_SETUP_RIGHT_HAND_SIDE = $(OBJECT_REALMATRIX) $(OBJECT_COMPLEXMATRIX)

SOURCE_SEGMENT_SETUP_SYSTEM_MATRIX = src/segment/setupSystemMatrix.cu
OBJECT_SEGMENT_SETUP_SYSTEM_MATRIX = obj/segment/setupSystemMatrix.o
DEPEND_SEGMENT_SETUP_SYSTEM_MATRIX = $(OBJECT_REALMATRIX)

SOURCE_SEGMENT_SOLVE_LINEAR_SYSTEM = src/segment/solveLinearSystem.cu
OBJECT_SEGMENT_SOLVE_LINEAR_SYSTEM = obj/segment/solveLinearSystem.o
DEPEND_SEGMENT_SOLVE_LINEAR_SYSTEM = $(OBJECT_REALMATRIX)

OBJECT_SEGMENT = $(OBJECT_SEGMENT_COMPUTE_FIELD_MATRICES) $(OBJECT_SEGMENT_COMPUTE_FIELD_VECTORS) $(OBJECT_SEGMENT_COMPUTE_FIELDS_FOR_LINEAR_SYSTEM) $(OBJECT_SEGMENT_SEGMENT) $(OBJECT_SEGMENT_SETUP) $(OBJECT_SEGMENT_SETUP_RIGHT_HAND_SIDE) $(OBJECT_SEGMENT_SETUP_SYSTEM_MATRIX) $(OBJECT_SEGMENT_SOLVE_LINEAR_SYSTEM)

SOURCE_COMPUTE_SUB_FIELDS = src/computeSubFields.cu
OBJECT_COMPUTE_SUB_FIELDS = obj/computeSubFields.o
DEPEND_COMPUTE_SUB_FIELDS = $(OBJECT_SEGMENT) $(OBJECT_GAUSSIAN_PROCESS) $(OBJECT_SEGMENT_COMPUTE_SCATTERED_SUB_FIELDS)

SOURCE_COMPUTE_REFLECTANCE = src/computeReflectance.cu
OBJECT_COMPUTE_REFLECTANCE = obj/computeReflectance.o
DEPEND_COMPUTE_REFLECTANCE = $(OBJECT_REALMATRIX)

SOURCE_BIOSCAT = src/BioScat.cpp
OBJECT_BIOSCAT = obj/BioScat.o
DEPEND_BIOSCAT = $(OBJECT_SEGMENT) $(OBJECT_GAUSSIAN_PROCESS) $(OBJECT_COMPUTE_SUB_FIELDS) $(OBJECT_COMPUTE_REFLECTANCE)

SOURCE_FORWARD = src/forward.cpp
OBJECT_FORWARD = obj/forward.o
DEPEND_FORWARD = $(OBJECT_COMPLEXMATRIX) $(OBJECT_BIOSCAT)

SOURCE_FAR_FIELD_PATTERN = src/farFieldPattern.cpp
OBJECT_FAR_FIELD_PATTERN = obj/farFieldPattern.o
DEPEND_FAR_FIELD_PATTERN = $(OBJECT_COMPLEXMATRIX) $(OBJECT_BIOSCAT) $(OBJECT_COMPUTE_FAR_FIELD_PATTERN)

SOURCE_GENERATE_ARTIFICIAL_DATA = src/generateArtificialData.cpp
OBJECT_GENERATE_ARTIFICIAL_DATA = obj/generateArtificialData.o
DEPEND_GENERATE_ARTIFICIAL_DATA = $(OBJECT_COMPLEXMATRIX) $(OBJECT_BIOSCAT) $(OBJECT_COMPUTE_REFLECTANCE_MATRIX) $(OBJECT_REALMATRIX)

OBJECTS_FORWARD = $(OBJECT_REALMATRIX) $(OBJECT_COMPLEXMATRIX) $(OBJECT_SEGMENT) $(OBJECT_GET_NANOSTRUCTURE) $(OBJECT_BIOSCAT) $(OBJECT_FORWARD) $(OBJECT_COMPUTE_SUB_FIELDS)

OBJECTS_FAR_FIELD_PATTERN = $(OBJECTS_FORWARD) $(OBJECT_FAR_FIELD_PATTERN)

SOURCE_CHOLESKY = src/GP/cholesky.cu
OBJECT_CHOLESKY = obj/GP/cholesky.o
DEPEND_CHOLESKY = 

SOURCE_COVARIANCE_MATRIX = src/GP/covariance_matrix.cu
OBJECT_COVARIANCE_MATRIX = obj/GP/covariance_matrix.o
DEPEND_COVARIANCE_MATRIX = 

SOURCE_CUDA_MALLOC_2D = src/GP/cudaMalloc2d.cu
OBJECT_CUDA_MALLOC_2D = obj/GP/cudaMalloc2d.o
DEPEND_CUDA_MALLOC_2D = 

SOURCE_REALISATION = src/GP/realisation.cu
OBJECT_REALISATION = obj/GP/realisation.o
DEPEND_REALISATION = 

SOURCE_GAUSSIAN_PROCESS_INNER = src/GP/gaussian_process_inner.cpp
OBJECT_GAUSSIAN_PROCESS_INNER = obj/GP/gaussian_process_inner.o
DEPEND_GAUSSIAN_PROCESS_INNER = 

SOURCE_COMPUTE_INVERSE = src/GP/compute_inverse.cu
OBJECT_COMPUTE_INVERSE = obj/GP/compute_inverse.o
DEPEND_COMPUTE_INVERSE = 

SOURCE_COMPUTE_PRIOR = src/GP/compute_prior.cu
OBJECT_COMPUTE_PRIOR = obj/GP/compute_prior.o
DEPEND_COMPUTE_PRIOR = 

SOURCE_GAUSSIAN_PROCESS = src/GP/GaussianProcess.cpp
OBJECT_GAUSSIAN_PROCESS = obj/GP/GaussianProcess.o
DEPEND_GAUSSIAN_PROCESS = $(OBJECT_CUDA_MALLOC_2D)

OBJECTS_GAUSSIAN_PROCESS = $(OBJECT_CHOLESKY) $(OBJECT_COVARIANCE_MATRIX) $(OBJECT_CUDA_MALLOC_2D) $(OBJECT_REALISATION) $(OBJECT_GAUSSIAN_PROCESS_INNER) $(OBJECT_GAUSSIAN_PROCESS) $(OBJECT_COMPUTE_INVERSE) $(OBJECT_COMPUTE_PRIOR)

SOURCE_INVERSE = src/inverse.cu
OBJECT_INVERSE = obj/inverse.o
DEPEND_INVERSE = $(OBJECT_COMPLEXMATRIX) $(OBJECT_BIOSCAT) $(OBJECTS_GAUSSIAN_PROCESS) $(OBJECT_COMPUTE_REFLECTANCE_MATRIX)

SOURCE_COMPUTE_REFLECTANCE_MATRIX = src/computeReflectanceMatrix.cu
OBJECT_COMPUTE_REFLECTANCE_MATRIX = obj/computeReflectanceMatrix.o
DEPEND_COMPUTE_REFLECTANCE_MATRIX = $(OBJECT_COMPLEXMATRIX) $(OBJECT_REALMATRIX)

# options and settings for the GCC compilers

CC		= gcc
CCC		= g++
CXX		= nvcc
DEFS	= 
OPT		= -g -O3
IPO		= 
ISA		= 
CHIP	= 
ARCH	= 
PARA	= -fopenmp -llapacke -llapack -lblas -lopenblas -lcusolver -lcublas -lcurand
XOPTS 	= -Xptxas=-v -arch=sm_90 -lineinfo -diag-suppress 2464 -diag-suppress 68  -diag-suppress 177 # Suppressing error regarding string conversion
CFLAGS	= $(DEFS) $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) -lstdc++
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)
LDFLAGS = -lm $(INCLUDES) $(SOFLAGS) $(XLIBS)
LIBS	= 
CUDA_PATH ?= /appl/cuda/12.2.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common  -D_MAKE_2D
SOFLAGS = -L$(CUDA_PATH)/lib64 
XLIBS =  -lcudart

all: so/forward.so so/gaussian_process.so so/generateArtificialData.so so/inverse.so so/farFieldPattern.so

so/forward.so: src/executeForward.c $(OBJECTS_FORWARD)
	$(CC) $(CFLAGS) $(INCLUDES)  -fPIC -shared -o $@ $^ $(SOFLAGS) $(LIBS)  -lcudart

so/farFieldPattern.so: src/executeFarFieldPattern.c $(OBJECTS_FAR_FIELD_PATTERN)
	$(CC) $(CFLAGS) $(INCLUDES)  -fPIC -shared -o $@ $^ $(SOFLAGS) $(LIBS)  -lcudart

so/inverse.so: src/executeInverse.c $(OBJECT_INVERSE)
	$(CC) $(CFLAGS) $(INCLUDES)  -fPIC -shared -o $@ $^ $(SOFLAGS) $(LIBS)  -lcudart

so/gaussian_process.so: src/GP/gaussian_process.c $(OBJECTS_GAUSSIAN_PROCESS)
	$(CC) $(CFLAGS) $(INCLUDES)  -fPIC -shared -o $@ $^ $(SOFLAGS) $(LIBS)  -lcudart

so/generateArtificialData.so: src/executeGenerateArtificialData.c $(OBJECTS_FORWARD) $(OBJECT_GENERATE_ARTIFICIAL_DATA)
	$(CC) $(CFLAGS) $(INCLUDES)  -fPIC -shared -o $@ $^ $(SOFLAGS) $(LIBS)  -lcudart

$(OBJECT_REALMATRIX): $(SOURCE_REALMATRIX) $(DEPEND_REALMATRIX) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_COMPLEXMATRIX): $(SOURCE_COMPLEXMATRIX) $(DEPEND_COMPLEXMATRIX) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_GET_NANOSTRUCTURE): $(SOURCE_GET_NANOSTRUCTURE) $(DEPEND_GET_NANOSTRUCTURE) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_COMPUTE_FIELD_MATRICES): $(SOURCE_SEGMENT_COMPUTE_FIELD_MATRICES) $(DEPEND_SEGMENT_COMPUTE_FIELD_MATRICES) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_COMPUTE_FAR_FIELD_PATTERN): $(SOURCE_SEGMENT_COMPUTE_FAR_FIELD_PATTERN) $(DEPEND_SEGMENT_COMPUTE_FAR_FIELD_PATTERN) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_COMPUTE_FAR_FIELD_PATTERN): $(SOURCE_COMPUTE_FAR_FIELD_PATTERN) $(DEPEND_COMPUTE_FAR_FIELD_PATTERN) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_COMPUTE_SCATTERED_SUB_FIELDS): $(SOURCE_SEGMENT_COMPUTE_SCATTERED_SUB_FIELDS) $(DEPEND_SEGMENT_COMPUTE_SCATTERED_SUB_FIELDS) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_COMPUTE_FIELD_VECTORS): $(SOURCE_SEGMENT_COMPUTE_FIELD_VECTORS) $(DEPEND_SEGMENT_COMPUTE_FIELD_VECTORS) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_COMPUTE_FIELDS_FOR_LINEAR_SYSTEM): $(SOURCE_SEGMENT_COMPUTE_FIELDS_FOR_LINEAR_SYSTEM) $(DEPEND_SEGMENT_COMPUTE_FIELDS_FOR_LINEAR_SYSTEM) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_SEGMENT): $(SOURCE_SEGMENT_SEGMENT) $(DEPEND_SEGMENT_SEGMENT) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_KERNELS): $(SOURCE_SEGMENT_KERNELS) $(DEPEND_SEGMENT_KERNELS) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_SETUP_1): $(SOURCE_SEGMENT_SETUP_1) $(DEPEND_SEGMENT_SETUP_1) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_SETUP_2): $(SOURCE_SEGMENT_SETUP_2) $(DEPEND_SEGMENT_SETUP_2) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_SETUP_3): $(SOURCE_SEGMENT_SETUP_3) $(DEPEND_SEGMENT_SETUP_3) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_SETUP): $(SOURCE_SEGMENT_SETUP) $(DEPEND_SEGMENT_SETUP) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_SETUP_RIGHT_HAND_SIDE): $(SOURCE_SEGMENT_SETUP_RIGHT_HAND_SIDE) $(DEPEND_SEGMENT_SETUP_RIGHT_HAND_SIDE) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_SETUP_SYSTEM_MATRIX): $(SOURCE_SEGMENT_SETUP_SYSTEM_MATRIX) $(DEPEND_SEGMENT_SETUP_SYSTEM_MATRIX) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_SEGMENT_SOLVE_LINEAR_SYSTEM): $(SOURCE_SEGMENT_SOLVE_LINEAR_SYSTEM) $(DEPEND_SEGMENT_SOLVE_LINEAR_SYSTEM) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_COMPUTE_SUB_FIELDS): $(SOURCE_COMPUTE_SUB_FIELDS) $(DEPEND_COMPUTE_SUB_FIELDS) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_COMPUTE_REFLECTANCE): $(SOURCE_COMPUTE_REFLECTANCE) $(DEPEND_COMPUTE_REFLECTANCE) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_COMPUTE_REFLECTANCE_MATRIX): $(SOURCE_COMPUTE_REFLECTANCE_MATRIX) $(DEPEND_COMPUTE_REFLECTANCE_MATRIX) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_BIOSCAT): $(SOURCE_BIOSCAT) $(DEPEND_BIOSCAT) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_FORWARD): $(SOURCE_FORWARD) $(DEPEND_FORWARD) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_FAR_FIELD_PATTERN): $(SOURCE_FAR_FIELD_PATTERN) $(DEPEND_FAR_FIELD_PATTERN) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_GENERATE_ARTIFICIAL_DATA): $(SOURCE_GENERATE_ARTIFICIAL_DATA) $(DEPEND_GENERATE_ARTIFICIAL_DATA) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_CHOLESKY): $(SOURCE_CHOLESKY) $(DEPEND_CHOLESKY) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_COVARIANCE_MATRIX): $(SOURCE_COVARIANCE_MATRIX) $(DEPEND_COVARIANCE_MATRIX) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_CUDA_MALLOC_2D): $(SOURCE_CUDA_MALLOC_2D) $(DEPEND_CUDA_MALLOC_2D) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_REALISATION): $(SOURCE_REALISATION) $(DEPEND_REALISATION) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_COMPUTE_INVERSE): $(SOURCE_COMPUTE_INVERSE) $(DEPEND_COMPUTE_INVERSE) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_COMPUTE_PRIOR): $(SOURCE_COMPUTE_PRIOR) $(DEPEND_COMPUTE_PRIOR) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_GAUSSIAN_PROCESS_INNER): $(SOURCE_GAUSSIAN_PROCESS_INNER) $(DEPEND_GAUSSIAN_PROCESS_INNER) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_GAUSSIAN_PROCESS): $(SOURCE_GAUSSIAN_PROCESS) $(DEPEND_GAUSSIAN_PROCESS) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

$(OBJECT_INVERSE): $(SOURCE_INVERSE) $(DEPEND_INVERSE) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart






#obj/segment/%.o: src/segment/%.cu $(DEPEND_SEGMENT)
#	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

#obj/segment/%.o: src/segment/%.cpp $(DEPEND_SEGMENT)
#	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

##obj/2D/%.o: src/2D/%.cu $(OBJECTS_MATRIX) 
##	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

##obj/2D/%.o: src/2D/%.cpp $(OBJECTS_MATRIX) 
##	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

##obj/3D/%.o: src/3D/%.cu
##	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

##obj/3D/%.o: src/3D/%.cpp
##	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

##obj/GP/%.o: src/GP/%.cu
##	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

##obj/GP/%.o: src/GP/%.cpp
##	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $^ -lcudart

## %.o: %.c
## 	$(CC) $(CFLAGS) -o $@ -c $<


clean:
	@/bin/rm -f core *.o *~
	@/bin/rm -f core obj/*.o
	@/bin/rm -f core obj/GP/*.o
	@/bin/rm -f core obj/2D/*.o
	@/bin/rm -f core obj/3D/*.o
	@/bin/rm -f core obj/segment/*.o
	@/bin/rm -f core obj/segment/setup/*.o
	

realclean: clean
	@/bin/rm -f 
	@/bin/rm -f core so/*.o
