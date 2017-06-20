 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel_divide(float *U, int k)
{
	// Thread index
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;

	// Block index
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;

	// Find position in Matrix
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;
	
	if (U[MATRIX_SIZE*k + k] == 0){
				printf("Numerical instability detected. The principal diagonal element is zero. \n");
			}
	if(column_number>k){
	double temp =  __ddiv_rd (U[MATRIX_SIZE*k + column_number] , U[MATRIX_SIZE * k + k]) ;
	U[MATRIX_SIZE*k + column_number] = (float)temp;
	 // Division step
	}
	__syncthreads();

}



__global__ void gauss_eliminate_kernel(float *U, int k)
{	// Thread index
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;
	// Block index
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;
	// Find position in Matrix
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;
	
    if(row_number==k){
	U[MATRIX_SIZE * k + k] = 1;	 // Division step
	}

	if(row_number>k&&column_number>k){
	double temp = __dsub_rd(U[MATRIX_SIZE*row_number +column_number] ,
			( __dmul_rd(U[MATRIX_SIZE*row_number + k] , U[MATRIX_SIZE * k + column_number])) ); // Elimination step
	U[MATRIX_SIZE * row_number + column_number] = (float)temp;
	}
	__syncthreads();

}

__global__ void gauss_eliminate_optimised_kernel(float *U, int k)
{
	// Thread index
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;

	// Block index
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;

	// Find position in Matrix
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;
	__shared__ float Rowsub[TILE_SIZE];
	__shared__ float Colsub[TILE_SIZE];
    if(row_number==k){
	U[MATRIX_SIZE * k + k] = 1;	 // Division step
	}
	__syncthreads();
	if(threadX==0 | threadY==0){
	Rowsub[threadX]=U[MATRIX_SIZE * k + column_number];
	Colsub[threadY]=U[MATRIX_SIZE*row_number + k];
	}
	__syncthreads();
	
	if(row_number>k&&column_number>k){
	double temp= (Rowsub[threadX] * Colsub[threadY]); // Elimination step
	U[MATRIX_SIZE * row_number + column_number] = U[MATRIX_SIZE*row_number +column_number] - (float)temp;
	}
	__syncthreads();
}

__global__ void gauss_eliminate_kernel_zero(float *U, int k)
{
	// Thread index
	int threadX = threadIdx.x; 
    int threadY = threadIdx.y;

	// Block index
	int blockX = blockIdx.x; 
    int blockY = blockIdx.y;

	// Find position in Matrix
	int column_number = TILE_SIZE * blockX + threadX; 
    int row_number = TILE_SIZE * blockY + threadY;
	  
	if(column_number < k+1 && row_number > k){
	U[MATRIX_SIZE * row_number + k] = 0;
	}
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
