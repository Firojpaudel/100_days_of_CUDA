#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256  // Define thread block size

// CUDA kernel for matrix-vector multiplication
__global__ void matVecMulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Get row index

    if (row < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += B[row * N + j] * C[j];  // Dot product
        }
        A[row] = sum;  // Store result
    }
}

// Host function
void matVecMul(float *A, float *B, float *C, int N) {
    // Allocate memory on device (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    matVecMulKernel<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 4;  // Example size
    float B[16] = {1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 10, 11, 12,
                   13, 14, 15, 16};  // 4x4 matrix
    float C[4] = {1, 1, 1, 1};  // 4-element vector
    float A[4];  // Output vector

    // Perform matrix-vector multiplication
    matVecMul(A, B, C, N);

    // Print result
    std::cout << "Resultant vector A:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
