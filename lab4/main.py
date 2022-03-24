from numba import cuda
import numpy as np


@cuda.jit
def matmul_gpu(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """

    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


# matrix multiplication using the cpu and C-like programming style (** strongly discouraged **)
def matmul_cpu(A, B):
    """Perform square matrix multiplication of C = A * B
    """
    C = np.zeros((A.shape[1], B.shape[1]))
    for i in range(A.shape[1]):
        for j in range(B.shape[0]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[i, k] * B[k, j]  # multiply elements in row i of A and column j of B and add to temp
            C[i, j] = tmp
    return C


# generate random vals
np.random.seed(42)
SIZE = 300
A = np.ones((SIZE, SIZE)).astype('float32')  # mat 1
B = np.ones((SIZE, SIZE)).astype('float32')  # mat 2
C = np.zeros((SIZE, SIZE)).astype('float32')  # mat where we store answer

d_A = cuda.to_device(A)  # Copy of A on the device
d_B = cuda.to_device(B)  # Copy of B on the device
d_C = cuda.device_array_like(A)  # malloc on the device

# data type
d_A.dtype

block = (16, 16)  # each block will contain 16x16 threads, typically 128 - 512 threads/block
grid_x = int(np.ceil(C.shape[0] / block[0]))
grid_y = int(np.ceil(C.shape[1] / block[1]))
grid = (grid_x, grid_y)  # we calculate the gridsize (number of blocks) from array
print(grid)
print(f"The kernel will be executed up to element {block[0] * grid_x}")

# execution of the kernel
matmul_gpu[grid, block](d_A, d_B, d_C)
cuda.synchronize()

C = d_C.copy_to_host()
print(C)

# execution of the python function in C-like programming style (strongly discouraged)
D = matmul_cpu(A, B)

# execution of the kernel
C = np.dot(A, B)

print(C)
