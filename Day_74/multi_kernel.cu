#include <cuda_runtime.h>

// Kernel 1: Compute inclusive prefix sums within each block and store block sums
__global__ void block_prefix_sum_kernel(const float* input, float* output, float* block_sums, unsigned int N) {
    extern __shared__ float temp[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input to shared memory
    temp[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Inclusive prefix sum within block using Kahan summation
    float sum = 0.0f;
    float compensation = 0.0f;
    for (unsigned int j = 0; j <= tid; j++) {
        float y = temp[j] - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    __syncthreads();

    // Store the block sum (last thread in block)
    if (tid == blockDim.x - 1 && i < N) {
        block_sums[blockIdx.x] = sum;
    }

    // Write block-local prefix sum to output
    if (i < N) {
        output[i] = sum;
    }
}

// Kernel 2: Compute inclusive prefix sum of block sums
__global__ void block_sums_prefix_sum_kernel(const float* block_sums, float* block_sums_output, unsigned int numBlocks) {
    extern __shared__ float temp[];

    unsigned int tid = threadIdx.x;

    // Load block sums into shared memory
    temp[tid] = (tid < numBlocks) ? block_sums[tid] : 0.0f;
    __syncthreads();

    // Inclusive prefix sum using Kahan summation
    float sum = 0.0f;
    float compensation = 0.0f;
    for (unsigned int j = 0; j <= tid; j++) {
        float y = temp[j] - compensation;
        float t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    __syncthreads();

    // Write result
    if (tid < numBlocks) {
        block_sums_output[tid] = sum;
    }
}

// Kernel 3: Add block sums to each block's elements
__global__ void add_block_sums_kernel(float* output, const float* block_sums_output, unsigned int N) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && i < N) {
        output[i] += block_sums_output[blockIdx.x - 1];
    }
}

extern "C" void solution(const float* input, float* output, size_t N) {
    const int threadsPerBlock = 1024;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate temporary array for block sums
    float* d_block_sums;
    cudaMalloc(&d_block_sums, numBlocks * sizeof(float));
    float* d_block_sums_output;
    cudaMalloc(&d_block_sums_output, numBlocks * sizeof(float));

    // Step 1: Compute per-block prefix sums
    block_prefix_sum_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, d_block_sums, static_cast<unsigned int>(N));

    // Step 2: Compute prefix sum of block sums
    if (numBlocks > 1) {
        block_sums_prefix_sum_kernel<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_block_sums, d_block_sums_output, numBlocks);

        // Step 3: Add block sums to output
        add_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(output, d_block_sums_output, static_cast<unsigned int>(N));
    }

    // Clean up
    cudaFree(d_block_sums);
    cudaFree(d_block_sums_output);
}