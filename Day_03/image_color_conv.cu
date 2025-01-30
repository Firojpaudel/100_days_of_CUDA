#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // This is needed for loading images 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  // This is needed for writing images(saving)

using namespace std;

#define CHANNELS 3  // RGB

__global__ void colorToGrayscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset];     
        unsigned char g = Pin[rgbOffset + 1]; 
        unsigned char b = Pin[rgbOffset + 2]; 

        Pout[grayOffset] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

int main() {
    int w, h, c;
    // Load the image
    unsigned char* h_in = stbi_load("pika.jpg", &w, &h, &c, CHANNELS);
    if (!h_in) { 
        cout << "Failed to load image.\n"; 
        return 1; 
    }

    // Allocate memory on the device
    unsigned char *d_in, *d_out, *h_out = new unsigned char[w * h];
    cudaMalloc(&d_in, w * h * CHANNELS);
    cudaMalloc(&d_out, w * h);
    cudaMemcpy(d_in, h_in, w * h * CHANNELS, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    // Launch the kernel
    colorToGrayscaleConversion<<<grid, block>>>(d_out, d_in, w, h);
    // Copy the result back to the host
    cudaMemcpy(h_out, d_out, w * h, cudaMemcpyDeviceToHost);

    // Save the output image
    stbi_write_png("output_pika.jpg", w, h, 1, h_out, w);
    cout << "Saved as output.png\n";

    // Free memory
    stbi_image_free(h_in);
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
