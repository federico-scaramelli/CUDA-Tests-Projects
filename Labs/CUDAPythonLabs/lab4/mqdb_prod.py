from numba import cuda
import numpy as np
import random


class mqdb:
    def __init__(self, mat, dim, n_blocks, dim_blocks):
        self.mat = mat
        self.n_blocks = n_blocks
        self.dim = dim
        self.dim_blocks = dim_blocks


@cuda.jit
def matmul_gpu(block_size, A, B, C, block_number):
    """Perform square matrix multiplication of C = A * B """

    row, col = cuda.grid(2)

    start_index = 0
    for i in range(block_number):
        start_index += block_size[i]
    if row < block_size[block_number] and col < block_size[block_number]:
        val = 0
        for k in range(block_size[block_number]):
            val += A[start_index + row, start_index + k] * B[start_index + k, start_index + col]
        C[start_index + row, start_index + col] = val


def build_mqdb(dim, n_blocks, val):
    if dim <= 0 or n_blocks <= 0 or dim <= n_blocks:
        print('Errore nella costruzione della matrice. Controllare i parametri.')
        return

    dim_blocks = list()
    block_sum = 0

    mat = np.zeros((dim, dim))

    for i in range(n_blocks - 1):
        dim_blocks.append(random.randint(1, dim - block_sum - (n_blocks - i)))
        mat[block_sum:block_sum + dim_blocks[i], block_sum:block_sum + dim_blocks[i]] = val
        block_sum += dim_blocks[i]

    dim_blocks.append(dim - block_sum)
    mat[block_sum:block_sum + dim_blocks[n_blocks - 1], block_sum:block_sum + dim_blocks[n_blocks - 1]] = val

    return mqdb(mat, dim, n_blocks, dim_blocks)


def build_matrix(source):
    return mqdb(source.mat, source.dim, source.n_blocks, source.dim_blocks)


# generate random vals
SIZE = 1000
N_BLOCKS = 30
A = build_mqdb(SIZE, N_BLOCKS, 1)  # mat 1
B = build_matrix(A)  # mat 2

d_A = cuda.to_device(A.mat)  # Copy of A on the device
d_B = cuda.to_device(B.mat)  # Copy of B on the device
d_block_size = cuda.to_device(A.dim_blocks)
d_C = cuda.device_array_like(A.mat)

# KERNEL PARAMETERS
block = (16, 16)  # each block will contain 16x16 threads, typically 128 - 512 threads/block

# execution of the kernel
for i in range(N_BLOCKS):
    grid_x = int(np.ceil(A.dim_blocks[i] / block[0]))
    grid_y = int(np.ceil(A.dim_blocks[i] / block[1]))
    grid = (grid_x, grid_y)  # we calculate the gridsize (number of blocks) from array
    print(grid)
    print(f"The kernel will be executed up to element {block[0] * grid_x}")
    matmul_gpu[grid, block](d_block_size, d_A, d_B, d_C, i)
    C = d_C.copy_to_host()
cuda.synchronize()

C = d_C.copy_to_host()
print(C)

# execution of the kernel
C1 = np.dot(A.mat, B.mat)
print(C1)

print(np.array_equal(C, C1))
