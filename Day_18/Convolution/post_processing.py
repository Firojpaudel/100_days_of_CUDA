## So this takes in the output convoluted bin file and shows the output

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read dimensions of the image (must match those used in the CUDA program)
height, width = 768, 1024  

# Load the output binary file
output_image = np.fromfile("output_image.bin", dtype=np.float32).reshape(height, width)

# Normalize and scale to [0, 255] for display
output_image = np.clip(output_image, 0, 1)  # Clamp values to [0, 1]
output_image = (output_image * 255).astype(np.uint8)

# Display the result using matplotlib
plt.imshow(output_image, cmap="gray")
plt.title("Convolution Result")
plt.axis("off")  # Hide axis
plt.show()

# Save the output image
cv2.imwrite("../images/Charmander_convoluted.png", output_image)
print("Image saved as output_image.png. Open it manually to view.")