
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void checkIndex(void) {
	typedef unsigned int uint;
	uint minus2 = 0;
	uint minus1 = 1;
	uint current = 0;
	uint sum = threadIdx.x + threadIdx.y + blockIdx.x + blockIdx.y;

	while (current <= sum) {
		if (current == sum) {
			printf("threadIdx:(%d, %d, %d) blockIdx:(%d, %d, %d) "
				"blockDim:(%d, %d, %d) gridDim:(%d, %d, %d)\nSum = %d; Fn-2 = %d; Fn-1 = %d;\n",
				threadIdx.x, threadIdx.y, threadIdx.z,
				blockIdx.x, blockIdx.y, blockIdx.z,
				blockDim.x, blockDim.y, blockDim.z,
				gridDim.x, gridDim.y, gridDim.z,
				sum, minus2, minus1);
			return;
		}
		minus2 = minus1;
		minus1 = current;
		current = minus1 + minus2;
	}
}

int main(int argc, char** argv) {

	// definisce grid e struttura dei blocchi
	dim3 block(10, 10);
	dim3 grid(20, 20);

	// controlla dim. dal lato host
	printf("CHECK lato host:\n");
	printf("grid.x = %d\t grid.y = %d\t grid.z = %d\n", grid.x, grid.y, grid.z);
	printf("block.x = %d\t block.y = %d\t block.z %d\n\n", block.x, block.y, block.z);

	// controlla dim. dal lato device
	printf("CHECK lato device:\n");
	checkIndex << <grid, block >> > ();

	// reset device
	cudaDeviceReset();
	return(0);
}