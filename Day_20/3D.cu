#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

using namespace std;

// Kernel function for 3D convolution
__global__ void convolution3D_basic_kernel(float *N, float *F, float *P, int r, int width, int height, int depth) {
    // Identify the output position
    int outCol = blockIdx.x * blockDim.x + threadIdx.x; // X-coordinate
    int outRow = blockIdx.y * blockDim.y + threadIdx.y; // Y-coordinate
    int outSlice = blockIdx.z * blockDim.z + threadIdx.z; // Z-coordinate (depth)

    // Ensure the thread operates within the bounds of the output matrix
    if (outCol < width && outRow < height && outSlice < depth) {
        float Pvalue = 0.0f;

        // Iterate through the 3D filter dimensions
        for (int fSlice = 0; fSlice < 2 * r + 1; fSlice++) {   // Depth of the filter
            for (int fRow = 0; fRow < 2 * r + 1; fRow++) {     // Height of the filter
                for (int fCol = 0; fCol < 2 * r + 1; fCol++) { // Width of the filter
                    int inSlice = outSlice - r + fSlice;
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;

                    // Check valid bounds for the input matrix
                    if (inSlice >= 0 && inSlice < depth && 
                        inRow >= 0 && inRow < height && 
                        inCol >= 0 && inCol < width) {
                        Pvalue += F[fSlice * (2 * r + 1) * (2 * r + 1) + fRow * (2 * r + 1) + fCol] * 
                                  N[inSlice * width * height + inRow * width + inCol];
                    }
                }
            }
        }

        // Write the computed value to the output matrix
        P[outSlice * width * height + outRow * width + outCol] = Pvalue;
    }
}

int main() {
    // Dimensions of input, filter, and output
    int width = 8, height = 8, depth = 8; // Input tensor size
    int r = 1; // Radius of the filter (filter size = (2r + 1)^3)
    int filterSize = 2 * r + 1;

    // Total sizes
    int N_size = width * height * depth; // Input size
    int F_size = filterSize * filterSize * filterSize; // Filter size
    int P_size = width * height * depth; // Output size (same as input for same padding)

    // Allocate host memory
    float *h_N = (float *)malloc(N_size * sizeof(float)); // Input tensor
    float *h_F = (float *)malloc(F_size * sizeof(float)); // Filter
    float *h_P = (float *)malloc(P_size * sizeof(float)); // Output tensor

    // Initialize input tensor and filter with some values
    for (int i = 0; i < N_size; i++) {
        h_N[i] = static_cast<float>(i % 10); // Example value
    }
    for (int i = 0; i < F_size; i++) {
        h_F[i] = 1.0f; // Example value
    }

    // Allocate device memory
    float *d_N, *d_F, *d_P;
    cudaMalloc((void **)&d_N, N_size * sizeof(float)); // Input tensor
    cudaMalloc((void **)&d_F, F_size * sizeof(float)); // Filter
    cudaMalloc((void **)&d_P, P_size * sizeof(float)); // Output tensor

    // Copy data from host to device
    cudaMemcpy(d_N, h_N, N_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, F_size * sizeof(float), cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions
    dim3 blockDim(4, 4, 4); // Number of threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y,
                 (depth + blockDim.z - 1) / blockDim.z); // Number of blocks in the grid

    // Launch the convolution kernel
    convolution3D_basic_kernel<<<gridDim, blockDim>>>(d_N, d_F, d_P, r, width, height, depth);

    // Copy the result back to the host
    cudaMemcpy(h_P, d_P, P_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result
    cout << "Output tensor:" << endl;
    for (int slice = 0; slice < depth; slice++) {
        cout << "Slice " << slice << ":" << endl;
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                cout << setw(6) << h_P[slice * width * height + row * width + col] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    // Free device memory
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);

    // Free host memory
    free(h_N);
    free(h_F);
    free(h_P);

    return 0;
}

