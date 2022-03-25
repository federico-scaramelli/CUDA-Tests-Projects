from numba import cuda
from numba.types import uint16
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


@cuda.jit
def BMP_hist(img, hist):
    row, col = cuda.grid(2)

    if row >= img.shape[0] or col >= img.shape[1]:
        return

    r = img[row, col, 0]
    g = img[row, col, 1]
    b = img[row, col, 2]
    cuda.atomic.add(hist, (0, r), 1)
    cuda.atomic.add(hist, (1, g), 1)
    cuda.atomic.add(hist, (2, b), 1)


img = Image.open('../lab2/images/dog.bmp')
img_rgb = img.convert('RGB')

# converto to numpy (host)
img_array = np.asarray(img)
print(f"Image size W x H x ch = {img_array.shape}")
print("Try to access a pixel channel: ", img_array[0, 0, 0])

# device data setup
d_I = cuda.to_device(img_array)
H = np.zeros((3, 256)).astype(np.float32)
d_H = cuda.to_device(H)

# KERNEL PARAMETERS
block = (16, 16)  # each block will contain 16x16 threads, typically 128 - 512 threads/block
grid_x = int(np.ceil(img_array.shape[0] / block[0]))
grid_y = int(np.ceil(img_array.shape[1] / block[1]))
grid = (grid_x, grid_y)  # we calculate the grid size (number of blocks) from array
print(grid)
print(f"The kernel will be executed up to element {block[0] * grid_x}")

# kernel launch
BMP_hist[grid, block](d_I, d_H)
hist = d_H.copy_to_host()
print(hist.shape)

plt.bar(np.arange(256), hist[0])
plt.title('Histogram (R)')
plt.show()
plt.bar(np.arange(256), hist[1])
plt.title('Histogram (G)')
plt.show()
plt.bar(np.arange(256), hist[2])
plt.title('Histogram (B)')
plt.show()