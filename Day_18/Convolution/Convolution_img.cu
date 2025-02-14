#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>

// CUDA kernel for 2D convolution
__global__ void conv2D(float *input, float *output, float *filter, int width, int height, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int r = filter_size / 2;
    float sum = 0.0;

    if (x < width && y < height) {
        for (int fy = -r; fy <= r; fy++) {
            for (int fx = -r; fx <= r; fx++) {
                int img_x = min(max(x + fx, 0), width - 1);  // Clamp x
                int img_y = min(max(y + fy, 0), height - 1); // Clamp y
                sum += input[img_y * width + img_x] * filter[(fy + r) * filter_size + (fx + r)];
            }
        }
        output[y * width + x] = sum;
    }
}

// Helper function to read binary data
void readBinaryFile(const char *filename, float *data, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    file.read(reinterpret_cast<char *>(data), size * sizeof(float));
    file.close();
}

// Helper function to write binary data
void writeBinaryFile(const char *filename, float *data, size_t size) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data), size * sizeof(float));
    file.close();
}

// Main function
int main() {
    // Read image dimensions manually (set these based on your input) 
    // To know that run.. 'prepare.py' first
    int width = 1024; 
    int height = 768; 
    int filter_size = 3;

    // Allocate memory for host
    size_t image_size = width * height * sizeof(float);
    size_t filter_size_bytes = filter_size * filter_size * sizeof(float);

    float *h_input = new float[width * height];
    float *h_output = new float[width * height];
    float *h_filter = new float[filter_size * filter_size];

    // Read input image and kernel from binary files
    readBinaryFile("input_image.bin", h_input, width * height);
    readBinaryFile("kernel.bin", h_filter, filter_size * filter_size);

    // Allocate memory for device
    float *d_input, *d_output, *d_filter;
    cudaMalloc(&d_input, image_size);
    cudaMalloc(&d_output, image_size);
    cudaMalloc(&d_filter, filter_size_bytes);

    // Copy host data to device
    cudaMemcpy(d_input, h_input, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size_bytes, cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    conv2D<<<gridSize, blockSize>>>(d_input, d_output, d_filter, width, height, filter_size);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost);

    // Write the output image to a binary file
    writeBinaryFile("output_image.bin", h_output, width * height);

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    delete[] h_input;
    delete[] h_output;
    delete[] h_filter;

    printf("Convolution complete: Output image saved to output_image.bin\n");
    return 0;
}
