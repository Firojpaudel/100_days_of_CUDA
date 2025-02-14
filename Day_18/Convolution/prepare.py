import cv2
import numpy as np

# Loading the image
image = cv2.imread("../images/Charmander.png", cv2.IMREAD_GRAYSCALE) ## Also converting to Grayscale 

# Normalizing the image (optional, for better numerical stability)
image = image.astype(np.float32) / 255.0

# Defining a 3x3 kernel (e.g., edge detection)
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=np.float32)

# Getting image dimensions
height, width = image.shape

# Saving the kernel and image to binary files to be read by CUDA
image.tofile("input_image.bin")
kernel.tofile("kernel.bin")

print(f"Image size: {height}x{width}")
