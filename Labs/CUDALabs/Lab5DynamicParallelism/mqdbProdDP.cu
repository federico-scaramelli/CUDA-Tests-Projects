#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../Utils/mqdb/mqdb.h"

#define BLOCK_SIZE 16     // block size


/*
 * Kernel for block sub-matrix product of mqdb
 */
__global__ void mqdbBlockProd(mqdb A, mqdb B, mqdb C, uint sdim, uint d, uint n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// jump to the right block sub-matrix
	int  offset = (n + 1) * sdim;

	// each thread computes an entry of the product matrix
	if ((row < d) && (col < d)) {
		float val = 0;

		for (int k = 0; k < d; k++)
			val += A.elem[row * n + k + offset] * B.elem[k * n + col + offset];
		C.elem[row * n + col + offset] = val;
	}
}

/*
 * Kernel for block sub-matrix product of mqdb: parent grid(1)
 */
__global__ void mqdbProdDP1(mqdb A, mqdb B, mqdb C, uint k, uint n) {

	uint offset = 0;
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
	
	for(int i = 0; i < k; i++)
	{
		uint submatrixDim = A.blkSize[i];
		//dim3 grid((submatrixDim + block.x - 1) / block.x, (submatrixDim + block.y - 1) / block.y);

		mqdbBlockProd <<<grid,block>>> (A, B, C, offset, submatrixDim, n);

		offset += submatrixDim;
	}
}

/*
 * Kernel for block sub-matrix product of mqdb: parent grid(k)
 */
__global__ void mqdbProdDPk(mqdb A, mqdb B, mqdb C, uint n) {
	
	uint submatrixDim = A.blkSize[threadIdx.x];

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((submatrixDim + block.x - 1) / block.x, (submatrixDim + block.y - 1) / block.y);

	uint offset = 0;
	for(int i = 0; i < threadIdx.x; i++)
	{
		offset += A.blkSize[i];
	}

	mqdbBlockProd << <grid, block >> > (A, B, C, offset, submatrixDim, n);
}