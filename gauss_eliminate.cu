#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include "gauss_eliminate_kernel.cu"

#define MIN_NUMBER 2
#define MAX_NUMBER 10

extern "C" int compute_gold(float*, const float*, unsigned int);
Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
void gauss_eliminate_on_device(const Matrix M, Matrix P);
void gauss_eliminate_on_device_optimised(const Matrix M, Matrix P);
int perform_simple_check(const Matrix M);
void print_matrix(const Matrix M);
void write_matrix_to_file(const Matrix M);
float get_random_number(int, int);
void checkCUDAError(const char *msg);
int checkResults(float *reference, float *gpu_result, int num_elements, float threshold);
float speedOnGPU ;

int 
main(int argc, char** argv) 
{
    // Matrices for the program
	Matrix  A; // The NxN input matrix
	Matrix  U; // The upper triangular matrix 
	Matrix  UO; // The upper triangular matrix 	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Allocate and initialize the matrices
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	U  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); 
	UO  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0); 
	
	// Perform Gaussian elimination on the CPU 
	Matrix reference = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	int status = compute_gold(reference.elements, A.elements, A.num_rows);
	gettimeofday(&stop, NULL);
	float speedOnCPU = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Execution time CPU = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	if(status == 0){
		printf("Failed to convert given matrix to upper triangular. Try again. Exiting. \n");
		exit(0);
	}
	status = perform_simple_check(reference); // Check that the principal diagonal elements are 1 
	if(status == 0){
		printf("The upper triangular matrix is incorrect. Exiting. \n");
		exit(0); 
	}
	printf("Gaussian elimination on the CPU was successful. \n");
	
	// Perform the vector-matrix multiplication on the GPU. Return the result in U
	gauss_eliminate_on_device(A, U);
   
	float speedup1 = (speedOnCPU)/(speedOnGPU);
 	printf("Speedup using Global memory = %0.4f\r\n",speedup1);
	
	// check if the device result is equivalent to the expected solution
	int num_elements = MATRIX_SIZE*MATRIX_SIZE;
        int res = checkResults(reference.elements, U.elements, num_elements, 0.01f);
        printf("Test Global %s\n", (1 == res) ? "PASSED" : "FAILED");
	print_matrix(reference);
	
	// Perform the vector-matrix multiplication on the GPU. Return the result in U
	gauss_eliminate_on_device_optimised(A, UO);
  
	speedup1 = (speedOnCPU)/(speedOnGPU);
 	printf("Speedup using shared memory = %0.4f\r\n",speedup1);
	
	// check if the device result is equivalent to the expected solution
	res = checkResults(reference.elements, UO.elements, num_elements, 0.01f);
        printf("Test Optimised %s\n", (1 == res) ? "PASSED" : "FAILED");
	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(U.elements); U.elements = NULL;
	free(UO.elements); UO.elements = NULL;
	free(reference.elements); reference.elements = NULL;
	return 0;
}


void 
gauss_eliminate_on_device(const Matrix A, Matrix U){
	Matrix Md = allocate_matrix_on_gpu(A);
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	copy_matrix_to_device(Md, A);
	// Setup the execution configuration
	
	dim3 threadsT(TILE_SIZE, 1);
	dim3 gridD((MATRIX_SIZE)/TILE_SIZE, 1);
	
	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 grid((MATRIX_SIZE)/TILE_SIZE, (MATRIX_SIZE)/TILE_SIZE);
   
	unsigned int k;
	//print_matrix(A);
	
	for (k = 0; k < MATRIX_SIZE; k++){
	
	gauss_eliminate_kernel_divide<<< gridD, threadsT >>>(Md.elements , k);
	cudaThreadSynchronize();
	
	gauss_eliminate_kernel<<< grid, threads >>>(Md.elements , k);
	cudaThreadSynchronize();
	
	gauss_eliminate_kernel_zero<<< grid, threads >>>(Md.elements , k);
	cudaThreadSynchronize();
	
	}

	copy_matrix_from_device(U, Md);
	print_matrix(U);
	gettimeofday(&stop, NULL);
	speedOnGPU = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Execution time GPU Global = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	}

void 
gauss_eliminate_on_device_optimised(const Matrix A, Matrix UO){
	Matrix Md = allocate_matrix_on_gpu(A);
	copy_matrix_to_device(Md, A);
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	// Setup the execution configuration
	
	dim3 threadsT(TILE_SIZE, 1);
	dim3 gridD((MATRIX_SIZE + TILE_SIZE - 1)/TILE_SIZE, 1);
	
	dim3 threads(TILE_SIZE, TILE_SIZE);
	dim3 grid((MATRIX_SIZE + TILE_SIZE - 1)/TILE_SIZE, (MATRIX_SIZE + TILE_SIZE - 1)/TILE_SIZE);
   
	unsigned int k;
	//print_matrix(A);
	
	for (k = 0; k < MATRIX_SIZE; k++){
	
	gauss_eliminate_kernel_divide<<< gridD, threadsT >>>(Md.elements , k);
	// execute the kernel
	cudaThreadSynchronize();
	
	gauss_eliminate_optimised_kernel<<< grid, threads >>>(Md.elements , k);
	cudaThreadSynchronize();
		
	gauss_eliminate_kernel_zero<<< grid, threads >>>(Md.elements , k);
	cudaThreadSynchronize();
	
	}
        gettimeofday(&stop, NULL);
	speedOnGPU = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
	printf("Execution time GPU Optimised = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	
	copy_matrix_from_device(UO, Md);
	//print_matrix(UO);
	
	}

// Allocate a device matrix of same size as M.
Matrix 
allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix 
allocate_matrix(int num_rows, int num_columns, int init){
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void 
copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void 
copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void 
print_matrix(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++){
				printf("Line number = %d ############## \n", i);
	for(unsigned int j = 0; j < M.num_columns; j++){

			printf("%f ", M.elements[i*M.num_rows + j]);
			}
		printf("\n");
	} 
	printf("\n");
	printf("####################################### \n");
}

// Returns a random floating-point number between the specified min and max values 
float 
get_random_number(int min, int max){
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}

// Performs a simple check on the upper triangular matrix. Checks to see if the principal diagonal elements are 1
int 
perform_simple_check(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++)
        if((fabs(M.elements[M.num_rows*i + i] - 1.0)) > 0.001) return 0;
	
    return 1;
} 

// Writes the matrix to a file 
void 
write_matrix_to_file(const Matrix M){
	FILE *fp;
	fp = fopen("matrix.txt", "wt");
	for(unsigned int i = 0; i < M.num_rows; i++){
        for(unsigned int j = 0; j < M.num_columns; j++)
            fprintf(fp, "%f", M.elements[i*M.num_rows + j]);
        }
    fclose(fp);
}

void 
checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}						 
}

int 
checkResults(float *reference, float *gpu_result, int num_elements, float threshold)
{
    int checkMark = 1;
    float epsilon = 0.0;
    
    for(int i = 0; i < num_elements; i++){
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > threshold){
            checkMark = 0;
			printf("error at %d \n",i);
			printf("element r %f and g %f \n",reference[i] ,gpu_result[i]);
            break;
        }
	}
	int maxEle;
    for(int i = 0; i < num_elements; i++){
        if(fabsf((reference[i] - gpu_result[i])/reference[i]) > epsilon){
            epsilon = fabsf((reference[i] - gpu_result[i])/reference[i]);
			maxEle=i;
		
        }
	}
    printf("Max epsilon = %f at i = %d value at cpu %f and gpu %f \n", epsilon,maxEle,reference[maxEle],gpu_result[maxEle]); 
    return checkMark;
}
