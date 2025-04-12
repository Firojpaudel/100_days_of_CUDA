#include <cuda_runtime.h>

// Kernel 1: Compute inclusive prefix sums within each block using Kogge-Stone
__global__ void block_prefix_sum_kernel(const float* input, float* output, float* block_sums, unsigned int N) {
    extern __shared__ float temp[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input to shared memory
    temp[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // Kogge-Stone inclusive prefix sum
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float t = 0.0f;
        if (tid >= stride) {
            t = temp[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            temp[tid] += t;
        }
        __syncthreads();
    }

    // Write block-local prefix sum to output
    if (i < N) {
        output[i] = temp[tid];
    }

    // Store block sum (last thread in block)
    if (tid == blockDim.x - 1 && blockIdx.x < gridDim.x) {
        block_sums[blockIdx.x] = temp[tid];
    }
}

// Kernel 2: Compute inclusive prefix sum of block sums using Kogge-Stone
__global__ void block_sums_prefix_sum_kernel(const float* block_sums, float* block_sums_output, unsigned int numBlocks) {
    extern __shared__ float temp[];

    unsigned int tid = threadIdx.x;

    // Load block sums into shared memory
    temp[tid] = (tid < numBlocks) ? block_sums[tid] : 0.0f;
    __syncthreads();

    // Kogge-Stone inclusive prefix sum
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        float t = 0.0f;
        if (tid >= stride) {
            t = temp[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            temp[tid] += t;
        }
        __syncthreads();
    }

    // Write result
    if (tid < numBlocks) {
        block_sums_output[tid] = temp[tid];
    }
}

// Kernel 3: Add block sums to each block’s elements
__global__ void add_block_sums_kernel(float* output, const float* block_sums_output, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x > 0 && i < N) {
        output[i] += block_sums_output[blockIdx.x - 1];
    }
}

extern "C" void solution(const float* input, float* output, size_t N) {
    const int threadsPerBlock = 1024; // Increased for better occupancy
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate temporary arrays
    float *d_block_sums, *d_block_sums_output;
    cudaMalloc(&d_block_sums, numBlocks * sizeof(float));
    cudaMalloc(&d_block_sums_output, numBlocks * sizeof(float));

    // Step 1: Compute per-block prefix sums
    block_prefix_sum_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, d_block_sums, static_cast<unsigned int>(N));

    // Step 2: Compute prefix sum of block sums
    if (numBlocks > 1) {
        int threadsForBlockSums = min(numBlocks, threadsPerBlock);
        block_sums_prefix_sum_kernel<<<1, threadsForBlockSums, threadsForBlockSums * sizeof(float)>>>(d_block_sums, d_block_sums_output, numBlocks);

        // Step 3: Add block sums to output
        add_block_sums_kernel<<<numBlocks, threadsPerBlock>>>(output, d_block_sums_output, static_cast<unsigned int>(N));
    }

    // Synchronize and clean up
    cudaDeviceSynchronize();
    cudaFree(d_block_sums);
    cudaFree(d_block_sums_output);
}