#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define FILTER_RADIUS 1              // For a 3x3 filter
#define TILE_SIZE 16                 // Tile (output) size
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)  // Filter width: 3

// Store the filter in constant memory (read-only for all threads)
__constant__ float d_filter[FILTER_WIDTH * FILTER_WIDTH];

// Kernel using caching for halo cells: shared memory holds only the internal tile.
__global__ void tiledConvCache(float *d_input, float *d_output, int width, int height) {
    // Each block processes a TILE_SIZE x TILE_SIZE region (output tile)
    // Allocate shared memory for the tile (without halo)
    __shared__ float sharedMem[TILE_SIZE][TILE_SIZE];

    // Global coordinates of the pixel this thread is responsible for
    int globalY = blockIdx.y * TILE_SIZE + threadIdx.y;
    int globalX = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Load the tile pixel from global memory into shared memory if within bounds
    if (globalY < height && globalX < width) {
        sharedMem[threadIdx.y][threadIdx.x] = d_input[globalY * width + globalX];
    } else {
        sharedMem[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Only compute the convolution if this thread corresponds to a valid output pixel
    if (globalY < height && globalX < width) {
        float sum = 0.0f;
        // Loop over the filter window (3x3)
        for (int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
            for (int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
                // Calculate neighbor's global coordinates
                int neighborY = globalY + dy;
                int neighborX = globalX + dx;

                float value;
                // Calculate the neighbor's local (shared memory) coordinates
                int localY = threadIdx.y + dy;
                int localX = threadIdx.x + dx;

                // If the neighbor is within this block's tile, use shared memory.
                // Otherwise, read from global memory (L2 cache will help here).
                if (localY >= 0 && localY < TILE_SIZE && localX >= 0 && localX < TILE_SIZE) {
                    value = sharedMem[localY][localX];
                } else {
                    if (neighborY >= 0 && neighborY < height && neighborX >= 0 && neighborX < width)
                        value = d_input[neighborY * width + neighborX];
                    else
                        value = 0.0f;  // Zero-padding for out-of-bound indices
                }
                // Multiply by the corresponding filter element (using constant memory)
                sum += value * d_filter[(dy + FILTER_RADIUS) * FILTER_WIDTH + (dx + FILTER_RADIUS)];
            }
        }
        // Write the result to the output image
        d_output[globalY * width + globalX] = sum;
    }
}

int main() {
    // Define image dimensions (e.g., 32x32) for demonstration.
    const int width = 32, height = 32;
    const int size = width * height * sizeof(float);

    // Allocate host memory for input and output images.
    float *h_input = new float[width * height];
    float *h_output = new float[width * height];

    // Initialize input image with a simple gradient pattern (value = i + j)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_input[i * width + j] = static_cast<float>(i + j);
        }
    }

    // Device memory allocation.
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define a Laplacian edge detection filter (3x3)
    // You can change these values if you prefer a different kernel.
    const float h_filter[FILTER_WIDTH * FILTER_WIDTH] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    cudaMemcpyToSymbol(d_filter, h_filter, sizeof(float) * FILTER_WIDTH * FILTER_WIDTH);

    // Define grid and block dimensions:
    // Each block covers a TILE_SIZE x TILE_SIZE region.
    dim3 blockDims(TILE_SIZE, TILE_SIZE);
    dim3 gridDims((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel with timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    tiledConvCache<<<gridDims, blockDims>>>(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Kernel execution time: " << ms << " ms\n";

    // Copy the output from device to host.
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print the filtered output image.
    cout << "Filtered Output Image:\n";
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            cout << h_output[i * width + j] << " ";
        }
        cout << "\n";
    }

    // Cleanup: free device and host memory.
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
