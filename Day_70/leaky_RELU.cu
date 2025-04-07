#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

// Leaky ReLU kernel
__global__ void leaky_relu_kernel(const float* input, float alpha, float* output, size_t M, size_t N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (i < M && j < N) {
        int idx = i * N + j;
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : alpha * val;
    }
}

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    // n = M (rows), m = N (columns)
    const int BLOCK_SIZE = 32; // 32x32 = 1024 threads, max for H100

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    leaky_relu_kernel<<<gridDim, blockDim>>>(input, alpha, output, n, m);
    cudaDeviceSynchronize();
}

// Test harness
int main() {
    // Test matrix size (e.g., matching ~9.8M elements from leaderboard)
    size_t M = 3136; // rows
    size_t N = 3136; // columns
    size_t size = M * N;
    float alpha = 0.01f; // Typical Leaky ReLU alpha

    // Host arrays
    float* h_input = new float[size];
    float* h_output = new float[size];

    // Initialize input with some test values
    for (size_t i = 0; i < size; i++) {
        h_input[i] = (i % 5 == 0) ? -1.0f * (i % 10) : 1.0f * (i % 10); // Mix of positive/negative
    }

    // Device arrays
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    // Run the solution
    solution(d_input, alpha, h_output, M, N);

    // Copy output back to host (note: h_output is currently a host pointer)
    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify a few values
    std::cout << "Sample results:" << std::endl;
    for (size_t i = 0; i < 5; i++) {
        size_t idx = i * N + i; // Diagonal elements
        float expected = (h_input[idx] > 0.0f) ? h_input[idx] : alpha * h_input[idx];
        std::cout << "Index " << idx << ": Input = " << h_input[idx]
                  << ", Expected = " << expected << ", Actual = " << h_output[idx]
                  << ", Diff = " << (expected - h_output[idx]) << std::endl;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}