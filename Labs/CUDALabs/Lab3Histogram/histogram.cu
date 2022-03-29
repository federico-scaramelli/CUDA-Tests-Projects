#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <limits.h>

#include "../Utils/bmp/ImageStuff.h"
#include "../Utils/bmp/bmpUtil.h"
#include "../Utils/common.h"

__global__ void histogramBMP(uint* hist, const pel* imgSrc, const uint w, const uint h) {

	uint threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIndex >= w * h)
		return;

	uint blockPerRow = (w + blockDim.x - 1) / blockDim.x;

	uint row = blockIdx.x / blockPerRow;
	uint col = threadIndex - row * w;

	uint bytePerRow = (w * 3 + 3) & (~3); //multiplo di 4

	uint sourceByte = bytePerRow * row + 3 * col;
	pel R = imgSrc[sourceByte];
	pel G = imgSrc[sourceByte + 1];
	pel B = imgSrc[sourceByte + 2];
	atomicAdd(&hist[R], 1);
	atomicAdd(&hist[G + 256], 1);
	atomicAdd(&hist[B + 512], 1);
}


void hist_CPU(uint* bins, const pel* imgSrc, const uint W, const uint H, const uint M) {
	for (int i = 0; i < W * H; i++) {
		uint r = i / W;              // row of the source pixel
		uint off = i - r * W;        // col of the source pixel

		//  ** byte granularity **
		uint p = M * r + 3 * off;      // src byte position of the pixel
		pel R = imgSrc[p];
		pel G = imgSrc[p + 1];
		pel B = imgSrc[p + 2];
		bins[R] += 1;
		bins[G + 256] += 1;
		bins[B + 512] += 1;
	}
}

int main(int argc, char** argv) {

	uint dimBlock = 1024;
	pel* imgBMP_CPU;     // Where images are stored in CPU
	pel* imgBMP_GPU;	 // Where images are stored in GPU

	uint* binsRGB_CPU, * binsRGB_GPU, * binsRGB_GPU2CPU;
	uint N_bins = 3 * 256;
	uint bin_size = N_bins * sizeof(uint);

	if (argc > 2)
		dimBlock = atoi(argv[2]);
	else if (argc < 2) {
		printf("\n\nUsage:  hist InputFilename dimBlock\n");
		exit(EXIT_FAILURE);
	}

	// bins for CPU & GPU
	binsRGB_CPU = (uint*)calloc(N_bins, sizeof(uint));
	binsRGB_GPU2CPU = (uint*)malloc(bin_size);
	CHECK(cudaMalloc((void**)&binsRGB_GPU, bin_size));

	// Create CPU memory to store the input image
	imgBMP_CPU = ReadBMPlin(argv[1]);
	if (imgBMP_CPU == NULL) {
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}

	// Allocate GPU buffer for image and bins
	CHECK(cudaMalloc((void**)&imgBMP_GPU, IMAGESIZE));

	// Copy input vectors from host memory to GPU buffers.
	CHECK(cudaMemcpy(imgBMP_GPU, imgBMP_CPU, IMAGESIZE, cudaMemcpyHostToDevice));

	// CPU histogram
	double start = clock() / (double)CLOCKS_PER_SEC;   // start time
	hist_CPU(binsRGB_CPU, imgBMP_CPU, WIDTH, HEIGHT, WIDTHB);
	double stop = clock() / (double)CLOCKS_PER_SEC;   // elapsed time
	printf("\nCPU elapsed time %f sec \n\n", stop - start);

	// invoke kernels (define grid and block sizes)
	uint nPixels = WIDTH * HEIGHT;
	int dimGrid = (nPixels + dimBlock - 1) / dimBlock;
	printf("\ndimGrid = %d   dimBlock = %d\n", dimGrid, dimBlock);

	start = clock() / (double)CLOCKS_PER_SEC;   // start time

	// TODO 
	histogramBMP << <dimGrid, dimBlock >> > (binsRGB_GPU, imgBMP_GPU, WIDTH, HEIGHT);

	CHECK(cudaDeviceSynchronize());
	stop = clock() / (double)CLOCKS_PER_SEC;   // elapsed time
	printf("\nGPU elapsed time %f sec \n\n", stop - start);

	// Copy output (results) from GPU buffer to host (CPU) memory.
	CHECK(cudaMemcpy(binsRGB_GPU2CPU, binsRGB_GPU, bin_size, cudaMemcpyDeviceToHost));

	for (int i = 0; i < N_bins / 3; i++)
		printf("bin_GPU[%d] = \t%d\t%d\t%d\t -- bin_CPU[%d] = \t%d\t%d\t%d\n", i,
			binsRGB_GPU2CPU[i], binsRGB_GPU2CPU[i + 256], binsRGB_GPU2CPU[i + 512],
			i, binsRGB_CPU[i], binsRGB_CPU[i + 256], binsRGB_CPU[i + 512]);

	// Deallocate GPU memory
	cudaFree(imgBMP_GPU);
	cudaFree(binsRGB_GPU);

	// tracing tools spel as Parallel Nsight and Visual Profiler to show complete traces.
	CHECK(cudaDeviceReset());

	return (EXIT_SUCCESS);
}


pel* ReadBMPlin(char* fn) {
	static pel* Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", fn);
		exit(EXIT_FAILURE);
	}

	pel HeaderInfo[54];
	size_t nByte = fread(HeaderInfo, sizeof(pel), 54, f); // read the 54-byte header
	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];
	img.width = width;
	int height = *(int*)&HeaderInfo[22];
	img.height = height;
	int RowBytes = (width * 3 + 3) & (~3);  // row is multiple of 4 pixel
	img.rowByte = RowBytes;
	//save header for re-use
	memcpy(img.headInfo, HeaderInfo, 54);
	printf("\n Input File name: %5s  (%d x %d)   File Size=%lu", fn, img.width, img.height, IMAGESIZE);

	// allocate memory to store the main image (1 Dimensional array)
	Img = (pel*)malloc(IMAGESIZE);
	if (Img == NULL)
		return Img;      // Cannot allocate memory
	// read the image from disk
	size_t out = fread(Img, sizeof(pel), IMAGESIZE, f);
	fclose(f);
	return Img;
}