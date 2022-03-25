from numba import cuda
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

@cuda.jit
def BMP_hist(...):


# image.save('beach1.bmp')
img = Image.open('GPUcomputing/images/dog.bmp')
img_rgb = img.convert('RGB')
img_rgb

# converto to numpy (host)
I = np.asarray(img)
print(f"Image size W x H x ch = {I.shape}")

# device data setup
d_I = cuda.to_device(I)
H = np.zeros((3,256)).astype(np.float32)
d_H = cuda.to_device(H)

#KERNEL PARAMETERS
block = (16, 16)  # each block will contain 16x16 threads, typically 128 - 512 threads/block
grid_x = int(np.ceil(I.shape[0] / block[0]))
grid_y = int(np.ceil(I.shape[1] / block[1]))
grid = (grid_x, grid_y)  # we calculate the gridsize (number of blocks) from array
print(grid)
print(f"The kernel will be executed up to element {block[0]*grid_x}")

# kernel launch
BMP_hist[grid, block](d_I, d_H)
hist = d_H.copy_to_host()
print(hist.shape)

plt.bar(np.arange(256),hist[0])
plt.title('Histogram (R)')
plt.show()
plt.bar(np.arange(256),hist[1])
plt.title('Histogram (G)')
plt.show()
plt.bar(np.arange(256),hist[2])
plt.title('Histogram (B)')
plt.show()