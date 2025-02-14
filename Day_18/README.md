## Summary of Day 18:

Today, I implemented convolution on a 2D image using a modular approach.

### Steps:

1. **Image Preparation**:
    - Loaded the image and converted it to binary files for the CUDA kernel.
    - The script `prepare.py` outputs two binary files: `kernel.bin` and `input_image.bin`.

2. **Image Convolution**:
    - Performed Laplace convolution for edge detection.
    - The CUDA file `Convolution_img.cu` processes the image and saves the result as `output_image.bin`.

3. **Displaying the Image**:
    - Used Matplotlib to display the convoluted image.
    - The final image is saved as `Charmander_convoluted.png`.

### Comparison:

| Input Image | Output Image |
|-------------|--------------|
| ![](./images/Charmander.png) | ![](./images/Charmander_convoluted.png) |

