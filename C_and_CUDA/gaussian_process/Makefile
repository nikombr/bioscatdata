# Makefile 

SOURCES	= src/cholesky.cu src/cudaMalloc2d.cu src/covariance_matrix.cu  src/realisation.cu src/gaussian_process_inner.cpp src/GaussianProcess.cpp
OBJECTS	= obj/cholesky.o obj/cudaMalloc2d.o obj/covariance_matrix.o obj/realisation.o  obj/gaussian_process_inner.o obj/GaussianProcess.o

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
PARA	= -fopenmp -llapack -lblas -lopenblas -lcusolver -lcublas -lcurand
XOPTS 	= -Xptxas=-v -arch=sm_90 -lineinfo -diag-suppress 2464 -diag-suppress 68  -diag-suppress 177 # Suppressing error regarding string conversion
CFLAGS	= $(DEFS) $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) -lstdc++
CXXFLAGS= --compiler-options "$(OPT) $(PARA)" $(XOPTS)
LDFLAGS = -lm $(INCLUDES) $(SOFLAGS) $(XLIBS)
LIBS	= 
CUDA_PATH ?= /appl/cuda/12.2.0
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/Common 
SOFLAGS = -L$(CUDA_PATH)/lib64 
XLIBS =  -lcudart

gaussian_process.so: src/gaussian_process.c $(OBJECTS) obj/gaussian_process_inner.o
	$(CC) $(CFLAGS) $(INCLUDES)  -fPIC -shared -o $@ $^ $(SOFLAGS) $(LIBS)  -lcudart




obj/%.o: src/%.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $< -lcudart

obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -Xcompiler -fPIC -shared -o $@  $< -lcudart

## %.o: %.c
## 	$(CC) $(CFLAGS) -o $@ -c $<


clean:
	@/bin/rm -f core *.o *~
	@/bin/rm -f core obj/*.o
	

realclean: clean
	@/bin/rm -f 
	@/bin/rm -f core gaussian_process.so
