
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "../Utils/common.h"

#define IDX(i,j,n) (i*n+j)
#define ABS(x,y) (x-y>=0?x-y:y-x)
#define N 1024
#define P 1024
#define M 1024
/*#define N 16
#define P 16
#define M 16*/

#define BLOCK_SIZE 16

/*
 * Kernel for matrix product with static SMEM
 *      C  =  A  *  B
 *    (NxM) (MxP) (PxM)
 */
__global__ void matProdSMEMstatic(float* A, float* B, float* C) {

	// TODO
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float A_subBlock[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float B_subBlock[BLOCK_SIZE][BLOCK_SIZE];

	float partialSum = 0.0;

	for (int i = 0; i < (P + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
		int c = i * BLOCK_SIZE + threadIdx.x;
		int r = i * BLOCK_SIZE + threadIdx.y;

		A_subBlock[threadIdx.y][threadIdx.x] = A[IDX(row, c, P)];
		B_subBlock[threadIdx.y][threadIdx.x] = B[IDX(r, col, M)];

		__syncthreads();

		for (int j = 0; j < BLOCK_SIZE; j++) {
			partialSum += A_subBlock[threadIdx.y][j] * B_subBlock[j][threadIdx.x];
		}

		__syncthreads();
	}

	if (row < N && col < M) {
		C[IDX(row, col, M)] = partialSum;
	}
}


/*
 * Kernel for matrix product using dynamic SMEM
 */
__global__ void matProdSMEMdynamic(float* A, float* B, float* C, const uint SMEMsize) {

	// SMEMsize = dimensione per due matrici (2*blocksize*blocksize) linearizzate

	// TODO
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	extern __shared__ float matrices[];
	float* A_subBlock = matrices;
	float* B_subBlock = &matrices[blockDim.x * blockDim.x];

	float partialSum = 0.0;

	for (int i = 0; i < (P + blockDim.x - 1) / blockDim.x; i++) {
		int c = i * blockDim.x + threadIdx.x;
		int r = i * blockDim.x + threadIdx.y;

		A_subBlock[IDX(threadIdx.y, threadIdx.x, blockDim.x)] = A[IDX(row, c, P)];
		B_subBlock[IDX(threadIdx.y, threadIdx.x, blockDim.x)] = B[IDX(r, col, M)];

		__syncthreads();

		for (int j = 0; j < blockDim.x; j++) {
			partialSum += A_subBlock[IDX(threadIdx.y, j, blockDim.x)] * B_subBlock[IDX(j, threadIdx.x, blockDim.x)];
		}

		__syncthreads();
	}

	if (row < N && col < M) {
		C[IDX(row, col, M)] = partialSum;
	}

}

/*
 * Kernel for naive matrix product
 */
__global__ void matProd(float* A, float* B, float* C) {
	// indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread computes an entry of the product matrix
	if ((row < N) && (col < M)) {
		float sum = 0;
		for (int k = 0; k < P; k++)
			sum += A[row * P + k] * B[k * M + col];
		C[row * M + col] = sum;
	}
}

/*
 *  matrix product on CPU
 */
void matProdCPU(float* A, float* B, float* C) {

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++) {
			float sum = 0;
			for (int k = 0; k < P; k++)
				sum += A[i * P + k] * B[k * M + j];
			C[i * M + j] = sum;
		}
}

/*
 * Test the device
 */
unsigned long testCUDADevice(void) {
	int dev = 0;

	cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
	cudaDeviceProp deviceProp;
	cudaSetDevice(dev);
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Device %d: \"%s\"\n", dev, deviceProp.name);
	printf("Total amount of shared memory available per block: %lu KB\n",
		deviceProp.sharedMemPerBlock / 1024);
	return deviceProp.sharedMemPerBlock;
}


/*
 * elementwise comparison between two mqdb
 */
void checkResult(float* A, float* B) {
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N * M; i++)
		if (ABS(A[i], B[i]) > epsilon) {
			match = 0;
			printf("   * Arrays do not match!\n");
			break;
		}
	if (match)
		printf("   Arrays match\n\n");
}

/*
 * MAIN
 */
int main(void) {
	// Kernels for matrix product
	//      C  =  A  *  B
	//    (NxM) (MxP) (PxM)
	uint rowA = N, rowB = P;
	uint colA = P, colB = M;
	uint rowC = N, colC = M;
	float* A, * B, * C, * C1;
	float* dev_A, * dev_B, * dev_C;

	// dims
	unsigned long Asize = rowA * colA * sizeof(float);
	unsigned long Bsize = rowB * colB * sizeof(float);
	unsigned long Csize = rowC * colC * sizeof(float);
	unsigned long maxSMEMbytes;
	uint nByteSMEM = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
	printf("N = %d, M = %d, P = %d\n", N, M, P);

	// test device shared memory
	maxSMEMbytes = testCUDADevice();
	if (maxSMEMbytes < nByteSMEM)
		printf("Shared memory usage WARNING: available: %lu, required: %d bytes\n",
			maxSMEMbytes, nByteSMEM);
	else
		printf("Total amount of shared memory required per block %.1f KB\n",
			(float)nByteSMEM / (float)1024);

	// malloc host memory
	A = (float*)malloc(Asize);
	B = (float*)malloc(Bsize);
	C = (float*)malloc(Csize);
	C1 = (float*)malloc(Csize);

	// malloc device memory
	CHECK(cudaMalloc((void**)&dev_A, Asize));
	CHECK(cudaMalloc((void**)&dev_B, Bsize));
	CHECK(cudaMalloc((void**)&dev_C, Csize));
	printf("Total amount of allocated memory on GPU %lu bytes\n\n",
		Asize + Bsize + Csize);

	// fill the matrices A and B
	for (int i = 0; i < N * P; i++)
		A[i] = rand() % 10;
	for (int i = 0; i < P * M; i++)
		B[i] = rand() % 10;
	matProdCPU(A, B, C);

	// copy matrices A and B to the GPU
	CHECK(cudaMemcpy(dev_A, A, Asize, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dev_B, B, Bsize, cudaMemcpyHostToDevice));

	/***********************************************************/
	/*              GPU matProdSMEM static SMEM               */
	/***********************************************************/
	// grid block dims = shared mem dims = BLOCK_SIZE
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	double start = seconds();
	matProdSMEMstatic << <grid, block >> > (dev_A, dev_B, dev_C);
	CHECK(cudaDeviceSynchronize());
	printf("   Kernel matProdSMEM static elapsed time GPU = %f\n", seconds() - start);

	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C1, dev_C, Csize, cudaMemcpyDeviceToHost));
	checkResult(C, C1);

	/***********************************************************/
	/*            GPU matProdSMEMD dynamic SMEM                */
	/***********************************************************/
	// set cache size
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	// try with various SMEM sizes
	uint sizes[] = { 8,16,32 };
	//uint sizes[] = { 8,16};
	//printSquareMatrix(C, M);
	for (int i = 0; i < 3; i++) {
		uint blockSize = sizes[i];
		block.x = blockSize;
		block.y = blockSize;
		grid.x = (M + block.x - 1) / block.x;
		grid.y = (N + block.y - 1) / block.y;
		uint SMEMsize = blockSize * blockSize;
		uint SMEMbyte = 2 * SMEMsize * sizeof(float);
		start = seconds();
		matProdSMEMdynamic << < grid, block, SMEMbyte >> > (dev_A, dev_B, dev_C, SMEMsize);
		CHECK(cudaDeviceSynchronize());
		printf("   Kernel matProdSMEM dynamic (SMEM size %d) elapsed time GPU = %f\n", blockSize, seconds() - start);

		// copy the array 'C' back from the GPU to the CPU
		CHECK(cudaMemcpy(C1, dev_C, Csize, cudaMemcpyDeviceToHost));
		checkResult(C, C1);
		//printf("\n\n");
		//printSquareMatrix(C1, M);
	}

	// free the memory allocated on the GPU
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);

	cudaDeviceReset();
	return EXIT_SUCCESS;
}
