#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 512  // Adjust based on shared memory constraints

// Kernel for performing local scan within each block
__global__ void localScan(int *d_in, int *d_out, int *d_sums, int N) {
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    // Load input into shared memory
    if (index < N)
        temp[tid] = d_in[index];
    else
        temp[tid] = 0;  // Padding

    __syncthreads();

    // Perform inclusive scan using Hillis-Steele algorithm
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write output
    if (index < N)
        d_out[index] = temp[tid];

    // Store last element of each block in d_sums for later second-level scan
    if (tid == blockDim.x - 1)
        d_sums[blockIdx.x] = temp[tid];
}

// Kernel for second-level scan on block sums
__global__ void scanBlockSums(int *d_sums, int *d_sums_scanned, int numBlocks) {
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;

    // Load into shared memory
    if (tid < numBlocks)
        temp[tid] = d_sums[tid];
    else
        temp[tid] = 0;

    __syncthreads();

    // Perform scan
    for (int offset = 1; offset < numBlocks; offset *= 2) {
        int val = 0;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    if (tid < numBlocks)
        d_sums_scanned[tid] = temp[tid];
}

// Kernel to add scanned sums back to each block
__global__ void addScannedSums(int *d_out, int *d_sums_scanned, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && index < N)
        d_out[index] += d_sums_scanned[blockIdx.x - 1];
}

// Function to perform Hierarchical Scan
void hierarchicalScan(int *h_in, int *h_out, int N) {
    int *d_in, *d_out, *d_sums, *d_sums_scanned;
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMalloc(&d_sums, numBlocks * sizeof(int));
    cudaMalloc(&d_sums_scanned, numBlocks * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA Timing Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: Perform local scans
    localScan<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, d_sums, N);
    cudaDeviceSynchronize();

    // Step 2: Perform scan on block sums
    scanBlockSums<<<1, BLOCK_SIZE>>>(d_sums, d_sums_scanned, numBlocks);
    cudaDeviceSynchronize();

    // Step 3: Add scanned sums back to each block
    addScannedSums<<<numBlocks, BLOCK_SIZE>>>(d_out, d_sums_scanned, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sums);
    cudaFree(d_sums_scanned);

    // Print execution time
    cout << "GPU Execution Time (Hierarchical Scan): " << milliseconds << " ms" << endl;
}

// Test the implementation
int main() {
    int N;
    cout << "Enter array size: ";
    cin >> N;

    int *h_in = new int[N];
    int *h_out = new int[N];

    cout << "Enter " << N << " elements: ";
    for (int i = 0; i < N; i++)
        cin >> h_in[i];

    hierarchicalScan(h_in, h_out, N);

    cout << "Prefix Sum Output: ";
    for (int i = 0; i < N; i++)
        cout << h_out[i] << " ";
    cout << endl;

    delete[] h_in;
    delete[] h_out;

    return 0;
}
