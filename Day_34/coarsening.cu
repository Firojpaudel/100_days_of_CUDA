#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 4  // Each block processes 4 elements per thread

// Kernel for Phase 1: Thread-local Sequential Scan
__global__ void local_scan(int *d_input, int *d_output, int *block_sums, int n) {
    __shared__ int shared_data[BLOCK_SIZE * 4];  // Shared memory for efficient access

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid * 4;  // Each thread processes 4 elements

    if (offset < n) {
        // Load data into shared memory
        for (int i = 0; i < 4 && (offset + i) < n; i++)
            shared_data[threadIdx.x * 4 + i] = d_input[offset + i];

        __syncthreads();

        // Sequential scan within the thread's subsection
        for (int i = 1; i < 4; i++)
            shared_data[threadIdx.x * 4 + i] += shared_data[threadIdx.x * 4 + i - 1];

        // Store results back
        for (int i = 0; i < 4 && (offset + i) < n; i++)
            d_output[offset + i] = shared_data[threadIdx.x * 4 + i];

        // Store last element for block-wide scan
        if (threadIdx.x == blockDim.x - 1)
            block_sums[blockIdx.x] = shared_data[threadIdx.x * 4 + 3];
    }
}

// Kernel for Phase 2: Block-wide Scan
__global__ void block_scan(int *block_sums, int *scanned_blocks, int num_blocks) {
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    if (tid < num_blocks)
        temp[tid] = block_sums[tid];

    __syncthreads();

    // Simple sequential scan in shared memory
    for (int i = 1; i < num_blocks; i++)
        if (tid == i)
            temp[tid] += temp[tid - 1];

    __syncthreads();

    if (tid < num_blocks)
        scanned_blocks[tid] = temp[tid];
}

// Kernel for Phase 3: Propagation
__global__ void propagate(int *d_output, int *scanned_blocks, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = tid * 4;

    if (blockIdx.x > 0 && offset < n) {
        int add_value = scanned_blocks[blockIdx.x - 1];
        for (int i = 0; i < 4 && (offset + i) < n; i++)
            d_output[offset + i] += add_value;
    }
}

int main() {
    int n;
    cout << "Enter array size: ";
    cin >> n;

    int *h_input = new int[n];
    int *h_output = new int[n];

    cout << "Enter " << n << " elements: ";
    for (int i = 0; i < n; i++)
        cin >> h_input[i];

    // Allocate device memory
    int *d_input, *d_output, *d_block_sums, *d_scanned_blocks;
    int num_blocks = (n + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_block_sums, num_blocks * sizeof(int));
    cudaMalloc(&d_scanned_blocks, num_blocks * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    // GPU Timing Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Phase 1: Local Scan
    local_scan<<<num_blocks, BLOCK_SIZE>>>(d_input, d_output, d_block_sums, n);
    cudaDeviceSynchronize();

    // Phase 2: Block-wide Scan
    block_scan<<<1, BLOCK_SIZE>>>(d_block_sums, d_scanned_blocks, num_blocks);
    cudaDeviceSynchronize();

    // Phase 3: Propagation
    propagate<<<num_blocks, BLOCK_SIZE>>>(d_output, d_scanned_blocks, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print output
    cout << "Prefix Sum Output: ";
    for (int i = 0; i < n; i++)
        cout << h_output[i] << " ";
    cout << endl;

    cout << "GPU Execution Time: " << milliseconds << " ms" << endl;

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_block_sums);
    cudaFree(d_scanned_blocks);

    return 0;
}
