#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"
#include "mqdb.h"

#define BLOCK_SIZE 16     // block size

struct tms {
	double CPUtms;
	double GPUtmsNaive;
	double GPUtmsMQDB;
	float density;
};


__global__ void matProd(mqdb A, mqdb B, mqdb C, int n) {
	// row & col indexes
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// each thread computes an entry of the product matrix
	if ((row < n) && (col < n)) {
		float val = 0;
		for (int k = 0; k < n; k++)
			val += A.elem[row * n + k] * B.elem[k * n + col];
		C.elem[row * n + col] = val;
	}
}


__global__ void mqdbBlockProd(mqdb A, mqdb B, mqdb C, int n, uint blockNumber) {

	// TODO
	uint row = blockIdx.y * blockDim.y + threadIdx.y;
	uint col = blockIdx.x * blockDim.x + threadIdx.x;

	//printf("Block Idx.x: %d; Block Idx.y: %d; \n", blockIdx.x, blockIdx.y);

	int startBlockIndex = 0;
	for (int i = 0; i < blockNumber; i++) {
		startBlockIndex += A.blkSize[i] * n + A.blkSize[i];
	}

	//printf("Start index: %d\n",startBlockIndex);

	// each thread computes an entry of the product matrix
	if ((row < A.blkSize[blockNumber]) && (col < A.blkSize[blockNumber])) {
		float val = 0;
		for (int k = 0; k < A.blkSize[blockNumber]; k++) {
			val += A.elem[startBlockIndex + row * n + k] * B.elem[startBlockIndex + k * n + col];
			// printf("MatA[%d] * MatB[%d] = %5.2f\n", startBlockIndex + row * n + k, startBlockIndex + k * n + col, A.elem[startBlockIndex + row * n + k] * B.elem[startBlockIndex + k * n + col]);
		}
		// printf("AGGIORNO MatC[%d] = %5.2f\n", startBlockIndex + row * n + col, val);
		C.elem[startBlockIndex + row * n + col] = val;
	}
}


void testKernelsMQDB(uint n, uint k, struct tms* times) {

	// mqdb host matrices
	mqdb A, B, C, C1, C2;

	// mqdb device matrices
	mqdb d_A, d_B, d_C;

	// fill in
	A = mqdbConst(n, k, 10, 1);
	B = mqdbConst(n, k, 10, 1);
	C = mqdbConst(n, k, 10, 1);
	C1 = mqdbConst(n, k, 10, 1);
	C2 = mqdbConst(n, k, 10, 1);

	ulong nBytes = n * n * sizeof(float);
	ulong kBytes = k * sizeof(uint);
	printf("Memory size required = %.1f (MB)\n", static_cast<float>(nBytes) / (1024.0 * 1024.0));

	// malloc and copy on device memory
	d_A.nBlocks = A.nBlocks;
	CHECK(cudaMalloc((void**)&d_A.blkSize, kBytes));
	CHECK(cudaMemcpy(d_A.blkSize, A.blkSize, kBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void**)&d_A.elem, nBytes));
	CHECK(cudaMemcpy(d_A.elem, A.elem, nBytes, cudaMemcpyHostToDevice));
	d_B.nBlocks = B.nBlocks;
	CHECK(cudaMalloc((void**)&d_B.blkSize, kBytes));
	CHECK(cudaMemcpy(d_B.blkSize, B.blkSize, kBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void**)&d_B.elem, nBytes));
	CHECK(cudaMemcpy(d_B.elem, B.elem, nBytes, cudaMemcpyHostToDevice));
	d_C.nBlocks = C.nBlocks;
	CHECK(cudaMalloc((void**)&d_C.blkSize, kBytes));
	CHECK(cudaMemcpy(d_C.blkSize, C.blkSize, kBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMalloc((void**)&d_C.elem, nBytes));
	CHECK(cudaMemset(d_C.elem, 0.0, nBytes));


	//                    CPU MQDB product
	printf("CPU MQDB product...\n");
	double start = seconds();
	mqdbProd(A, B, C);
	double CPUTime = seconds() - start;
	printf("   CPU elapsed time: %.5f (sec)\n\n", CPUTime);


	//                     GPU mat product 
	printf("Kernel (naive) mat product...\n");
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
	start = seconds();
	matProd <<<grid, block >>> (d_A, d_B, d_C, n);
	CHECK(cudaDeviceSynchronize());
	double GPUtime1 = seconds() - start;
	printf("   elapsed time:                %.2f (sec)\n", GPUtime1);
	printf("   speedup vs CPU MQDB product: %.2f\n", CPUTime / GPUtime1);
	CHECK(cudaMemcpy(C1.elem, d_C.elem, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaMemset(d_C.elem, 0.0, nBytes));
	checkResult(C, C1);
	//	mqdbDisplay(C1);
	
	//                     GPU MQDB product
	printf("Kernel MQDB product...\n");

	// TODO
	start = seconds();

	for (uint i = 0; i < A.nBlocks; i++) {
		grid.x = (A.blkSize[i] + block.x - 1) / block.x;
		grid.y = (A.blkSize[i] + block.y - 1) / block.y;
		mqdbBlockProd <<<grid, block >>> (d_A, d_B, d_C, n, i);
	}
	CHECK(cudaDeviceSynchronize());
	double GPUtime2 = seconds() - start;
	printf("   elapsed time:                    %.2f (sec)\n", GPUtime2);
	printf("   speedup vs CPU MQDB product:     %.2f\n", CPUTime / GPUtime2);
	printf("   speedup vs GPU std mat product:  %.2f\n", GPUtime1 / GPUtime2);
	// copy the array 'C' back from the GPU to the CPU
	CHECK(cudaMemcpy(C2.elem, d_C.elem, nBytes, cudaMemcpyDeviceToHost));
	CHECK(cudaMemset(d_C.elem, 0.0, nBytes));
	// mqdbDisplay(&C);
	// mqdbDisplay(&C2);
	checkResult(C, C2);

	CHECK(cudaFree(d_A.elem));
	CHECK(cudaFree(d_B.elem));
	CHECK(cudaFree(d_C.elem));

	// collect times
	times->CPUtms = CPUTime;
	times->GPUtmsNaive = GPUtime1;
	times->GPUtmsMQDB = GPUtime2;

	float den = 0;
	for (uint j = 0; j < k; j++)
		den += A.blkSize[j] * A.blkSize[j];
	times->density = den / (n * n);
}

int main(int argc, char* argv[]) {
	uint n = 2*1024;      // matrix size
	const uint min_k = 30;       // max num of blocks
	const uint max_k = 30;       // max num of blocks
	// uint n = 30;      // matrix size
	// const uint min_k = 2;       // max num of blocks
	// const uint max_k = 2;       // max num of blocks

	struct tms times[max_k - min_k + 1];

	// multiple tests on kernels
	for (uint k = min_k; k <= max_k; k++) {
		printf("\n*****   k = %d --- (avg block size = %f)\n", k, static_cast<float>(n) / k);
		testKernelsMQDB(n, k, &times[k - min_k]);
	}

	FILE* fd;
	fd = fopen("res.csv", "w");
	if (fd == NULL) {
		perror("file error!\n");
		exit(1);
	}

	// write results on file
	fprintf(fd, "num blocks,");
	for (uint j = 0; j <= max_k - min_k; j++)
		fprintf(fd, "%d,", j + min_k);

	fprintf(fd, "\nCPU MQDB product,");
	for (uint j = 0; j <= max_k - min_k; j++)
		fprintf(fd, "%.4f,", times[j].CPUtms);

	fprintf(fd, "\nKernel mat product naive,");
	for (uint j = 0; j <= max_k - min_k; j++)
		fprintf(fd, "%.4f,", times[j].GPUtmsNaive);

	fprintf(fd, "\nKernel MQDB product,");
	for (uint j = 0; j <= max_k - min_k; j++)
		fprintf(fd, "%.4f,", times[j].GPUtmsMQDB);

	fprintf(fd, "\ndensity,");
	for (uint j = 0; j <= max_k - min_k; j++)
		fprintf(fd, "%.4f,", times[j].density);

	fclose(fd);

	return 0;
}
