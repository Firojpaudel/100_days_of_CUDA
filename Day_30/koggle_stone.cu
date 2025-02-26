#include <iostream>
#include <cuda.h>

#define N 8  // Array size (must be a power of 2 for simplicity)
#define BLOCK_SIZE 8  // Number of threads per block (same as N for simplicity)

// CUDA Kernel for Kogge-Stone Scan (Prefix Sum)
__global__ void kogge_stone_scan(int *d_arr, int *d_result) {
    __shared__ int temp[N];  // Shared memory for intra-block communication

    int tid = threadIdx.x;  // Thread ID within the block

    // Load input into shared memory
    temp[tid] = d_arr[tid];
    __syncthreads();  // Ensure all threads have loaded data

    // **Up-Sweep (Reduction Phase)**
    for (int step = 1; step < N; step *= 2) {
        int val = 0;
        if (tid >= step) {
            val = temp[tid - step];  // Get value from the left
        }
        __syncthreads();  // Ensure all reads are done before writing
        temp[tid] += val;
        __syncthreads();  // Ensure all writes are done before next step
    }

    // **Exclusive Scan Fix: Shift Right**
    if (tid == 0) temp[tid] = 0;  // First element should be zero
    else temp[tid] = temp[tid - 1];

    __syncthreads();  // Synchronize before writing to global memory

    // Store the result back in the output array
    d_result[tid] = temp[tid];
}

// Host function
int main() {
    int h_arr[N] = {3, 1, 7, 0, 4, 1, 6, 3};  // Input array
    int h_result[N];  // Output array for prefix sum

    int *d_arr, *d_result;  // Device pointers

    // Allocate device memory
    cudaMalloc((void**)&d_arr, N * sizeof(int));
    cudaMalloc((void**)&d_result, N * sizeof(int));

    // Copy input array from host to device
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel (1 block with N threads)
    kogge_stone_scan<<<1, BLOCK_SIZE>>>(d_arr, d_result);

    // Copy result back to host
    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Prefix Sum: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_result);

    return 0;
}
