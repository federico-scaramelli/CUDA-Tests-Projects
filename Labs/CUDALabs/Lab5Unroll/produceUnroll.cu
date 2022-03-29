
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "main.cu"
#include "G:/Progetti/CUDA/lab3/Lab3/Lab3/common.h"

/*
 *  Device function: block parallel reduction based on warp unrolling
 */
__device__ void blockWarpUnroll(int* thisBlock, int blockDim, uint tid) {
    // in-place reduction in global memory
    for (int stride = blockDim / 2; stride > 32; stride >>= 1) {
        if (tid < stride)
            thisBlock[tid] += thisBlock[tid + stride];

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32) {
        volatile int* vmem = thisBlock;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
}

/*
 *  Block by block parallel implementation with warp unrolling
 */
__global__ void blockParReduceUroll(int* in, int* out, ulong n) {

    uint tid = threadIdx.x;
    ulong idx = blockIdx.x * blockDim.x + threadIdx.x;

    // boundary check
    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    int* thisBlock = in + blockIdx.x * blockDim.x;

    // block parall. reduction based on warp unrolling 
    blockWarpUnroll(thisBlock, blockDim.x, tid);

    // write result for this block to global mem
    if (tid == 0)
        out[blockIdx.x] = thisBlock[0];
}

/*
 *  Multi block parallel implementation with block and warp unrolling
 */
__global__ void multBlockParReduceUroll8(int* in, int* out, ulong n) {

    uint threadId = threadIdx.x;

    ulong idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    // boundary check
    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    //offset puntatore array del gruppo di 8 blocks che devono essere ridotti a 1 block
    int* thisBlock = in + blockIdx.x * blockDim.x * 8;

    if (idx + 7 * blockDim.x < n)
    {
        int a1 = in[idx];
        int a2 = in[idx + 1 * blockDim.x];
        int a3 = in[idx + 2 * blockDim.x];
        int a4 = in[idx + 3 * blockDim.x];
        int a5 = in[idx + 4 * blockDim.x];
        int a6 = in[idx + 5 * blockDim.x];
        int a7 = in[idx + 6 * blockDim.x];
        int a8 = in[idx + 7 * blockDim.x];
        in[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    //UNROLLING 8 BLOCKS

    // block parall. reduction based on warp unrolling 
    blockWarpUnroll(thisBlock, blockDim.x, threadId);

    // write result for this block to global mem
    if (threadId == 0)
        out[blockIdx.x] = thisBlock[0];
}

/*
 *  Multi block parallel implementation with block and warp unrolling
 */
__global__ void multBlockParReduceUroll16(int* in, int* out, ulong n) {

    uint threadId = threadIdx.x;

    ulong idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;
    // boundary check
    if (idx >= n)
        return;

    // convert global data pointer to the local pointer of this block
    //offset puntatore array del gruppo di 8 blocks che devono essere ridotti a 1 block
    int* thisBlock = in + blockIdx.x * blockDim.x * 16;

    if (idx + 15 * blockDim.x < n)
    {
        int a1 = in[idx];
        int a2 = in[idx + 1 * blockDim.x];
        int a3 = in[idx + 2 * blockDim.x];
        int a4 = in[idx + 3 * blockDim.x];
        int a5 = in[idx + 4 * blockDim.x];
        int a6 = in[idx + 5 * blockDim.x];
        int a7 = in[idx + 6 * blockDim.x];
        int a8 = in[idx + 7 * blockDim.x];
        int a9 = in[idx + 8 * blockDim.x];
        int a10 = in[idx + 9 * blockDim.x];
        int a11 = in[idx + 10 * blockDim.x];
        int a12 = in[idx + 11 * blockDim.x];
        int a13 = in[idx + 12 * blockDim.x];
        int a14 = in[idx + 13 * blockDim.x];
        int a15 = in[idx + 14 * blockDim.x];
        int a16 = in[idx + 15 * blockDim.x];
        in[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15 + a16;
    }
    __syncthreads();

    //UNROLLING 16 BLOCKS

    // block parall. reduction based on warp unrolling 
    blockWarpUnroll(thisBlock, blockDim.x, threadId);

    // write result for this block to global mem
    if (threadId == 0)
        out[blockIdx.x] = thisBlock[0];

}