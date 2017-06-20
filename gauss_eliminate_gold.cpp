#include <stdio.h>
#include <stdlib.h>
extern "C" int compute_gold( float*, const float*, unsigned int);

// Compute reference data set
int 
compute_gold(float* U, const float* A, unsigned int num_elements){
	unsigned int i, j, k;
	
	// Copy the contents of the A matrix into the U matrix
    for (i = 0; i < num_elements; i ++)
		for(j = 0; j < num_elements; j++)
			U[num_elements * i + j] = A[num_elements*i + j];

	// Perform Gaussian elimination in place on the U matrix
	for (k = 0; k < num_elements; k++){
		for (j = (k + 1); j < num_elements; j++){ // Reduce the current row
			if (U[num_elements*k + k] == 0){
				printf("Numerical instability detected. The principal diagonal element is zero. \n");
				return 0;
			}
            U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]); // Division step
		}
		
        U[num_elements * k + k] = 1; // Set the principal diagonal entry in U to be 1 
        for (i = (k+1); i < num_elements; i++){
			for (j = (k+1); j < num_elements; j++)
				U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]); // Elimination step
			
			U[num_elements * i + k] = 0; 
		} 
	}	
	return 1;
}
