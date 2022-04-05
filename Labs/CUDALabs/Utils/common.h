#ifndef COMMON_H
#define COMMON_H

#include <cstdio>

#include "cuda_runtime.h"
#include <ctime>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

// It prints the linear array of the given size; for debugging purposes.
inline void printLinearArray(int* array, int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        if (i != 0) printf(", ");
        printf("%d", array[i]);
    }
    printf("]");
}

// It prints the square matrix of the given edge length; for debugging purposes.
inline void printSquareMatrix(float* matrix, int matrixEdge) {
    for (int row = 0; row < matrixEdge; row++) {
        for (int col = 0; col < matrixEdge; col++) {
            int index = row * matrixEdge + col;
            if (col != 0) printf(" ");
            printf("%.2f", matrix[index]);
        }
        if (row != matrixEdge - 1) printf("\n");
    }
}

// inline double seconds() {
//     struct timeval tp;
//     int i = gettimeofday(&tp);
//     return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
// }

inline double seconds() {
    return clock() / (double)CLOCKS_PER_SEC;
}

inline void device_name() {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));
}

typedef unsigned long ulong;
typedef unsigned int uint;

#endif // _COMMON_H