// All includes and defines similar to yesterdays code
#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BLUR_SIZE 8 // The size of the blur filter
#define CHANNELS 3  // RGB channels

using namespace std;

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixValR = 0, pixValG = 0, pixValB = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    int offset = (curRow * w + curCol) * CHANNELS;
                    pixValR += in[offset];
                    pixValG += in[offset + 1];
                    pixValB += in[offset + 2];
                    ++pixels;
                }
            }
        }

        int offset = (row * w + col) * CHANNELS;
        out[offset] = pixValR / pixels;
        out[offset + 1] = pixValG / pixels;
        out[offset + 2] = pixValB / pixels;
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
    unsigned char *d_in, *d_out, *h_out = new unsigned char[w * h * CHANNELS];
    cudaMalloc(&d_in, w * h * CHANNELS);
    cudaMalloc(&d_out, w * h * CHANNELS);
    cudaMemcpy(d_in, h_in, w * h * CHANNELS, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    // Launch the kernel
    blurKernel<<<grid, block>>>(d_in, d_out, w, h);
    // Copy the result back to the host
    cudaMemcpy(h_out, d_out, w * h * CHANNELS, cudaMemcpyDeviceToHost);

    // Save the output image
    stbi_write_png("blurred_output.png", w, h, CHANNELS, h_out, w * CHANNELS);
    cout << "Blurred image saved as blurred_output.png\n";

    // Free memory
    stbi_image_free(h_in);
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}