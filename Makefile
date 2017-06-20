# A simple CUDA makefile
#
# Author: Naga Kandasamy
# Date: 9/16/2015
#
# CUDA depends on two things:
#  1) The CUDA nvcc compiler, which needs to be on your path,
#	or called directly, which we do here
#  2) The CUDA shared library being available at runtime,
#	which we make available by setting the LD_LIBRARY_PATH
#	variable for the duration of the makefile.
#   
#   Note that you can set your PATH and LD_LIBRARY_PATH variables as part of your
# .bash_profile so that you can compile and run without using this makefile.

NVCCFLAGS	:= -O3 -gencode arch=compute_30,code=sm_30 -prec-div=true
NVCC		:= /usr/local/cuda/bin/nvcc
LD_LIBRARY_PATH	:= /usr/local/cuda/lib64

all: gauss_eliminate

gauss_eliminate: gauss_eliminate.cu gauss_eliminate_gold.cpp 
	$(NVCC) -o gauss_eliminate gauss_eliminate.cu gauss_eliminate_gold.cpp $(NVCCFLAGS)

clean:
	rm gauss_eliminate 
