#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

/**
 * Kernel to compute cumulative products within each block and store block products.
 * Each thread loads 4 elements into shared memory, then thread 0 computes the
 * cumulative product within the block. Results are written to output, and the
 * block product is stored for inter-block computation.
 */
__global__ void cumProdBlockKernel(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  float* __restrict__ block_products, 
                                  size_t n) {
    __shared__ float shared[BLOCK_SIZE * ELEMENTS_PER_THREAD];
    
    size_t tid = threadIdx.x;
    size_t block_start = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD;
    size_t idx = block_start + tid * ELEMENTS_PER_THREAD;
    
    // Load 4 floats per thread into shared memory using vectorized access
    float4 vals;
    if (idx + 3 < n) {
        vals = *reinterpret_cast<const float4*>(&input[idx]);
    } else {
        // Handle boundary: use 1.0f for elements beyond n
        vals.x = (idx + 0 < n) ? input[idx + 0] : 1.0f;
        vals.y = (idx + 1 < n) ? input[idx + 1] : 1.0f;
        vals.z = (idx + 2 < n) ? input[idx + 2] : 1.0f;
        vals.w = (idx + 3 < n) ? input[idx + 3] : 1.0f;
    }
    
    shared[tid * ELEMENTS_PER_THREAD + 0] = vals.x;
    shared[tid * ELEMENTS_PER_THREAD + 1] = vals.y;
    shared[tid * ELEMENTS_PER_THREAD + 2] = vals.z;
    shared[tid * ELEMENTS_PER_THREAD + 3] = vals.w;
    __syncthreads();
    
    // Compute prefix product within block using thread 0
    if (threadIdx.x == 0) {
        float temp = shared[0];
        for (size_t i = 1; i < BLOCK_SIZE * ELEMENTS_PER_THREAD; ++i) {
            if (block_start + i < n) {
                temp *= shared[i];
                shared[i] = temp;
            }
        }
    }
    __syncthreads();
    
    // Write results to output using vectorized access where possible
    if (idx + 3 < n) {
        float4 result;
        result.x = shared[tid * ELEMENTS_PER_THREAD + 0];
        result.y = shared[tid * ELEMENTS_PER_THREAD + 1];
        result.z = shared[tid * ELEMENTS_PER_THREAD + 2];
        result.w = shared[tid * ELEMENTS_PER_THREAD + 3];
        *reinterpret_cast<float4*>(&output[idx]) = result;
    } else {
        for (size_t i = 0; i < ELEMENTS_PER_THREAD && idx + i < n; ++i) {
            output[idx + i] = shared[tid * ELEMENTS_PER_THREAD + i];
        }
    }
    
    // Store the block product for inter-block propagation
    if (tid == 0) {
        size_t last_valid = block_start + BLOCK_SIZE * ELEMENTS_PER_THREAD;
        if (last_valid > n) last_valid = n;
        if (last_valid > block_start) {
            size_t last_idx = last_valid - block_start - 1;
            block_products[blockIdx.x] = shared[last_idx];
        } else {
            block_products[blockIdx.x] = 1.0f;
        }
    }
}

/**
 * Kernel to compute the cumulative product of block_products array.
 * Uses a single thread for simplicity, suitable since num_blocks is typically small.
 */
__global__ void computeCumBlockProducts(float* block_products, size_t num_blocks) {
    if (threadIdx.x == 0) {
        for (size_t i = 1; i < num_blocks; ++i) {
            block_products[i] *= block_products[i - 1];
        }
    }
}

/**
 * Kernel to propagate cumulative block products to the output array.
 * Each thread multiplies its segment by the cumulative product of all previous blocks.
 */
__global__ void cumProdPropagateKernel(float* __restrict__ output, 
                                      const float* __restrict__ block_products, 
                                      size_t n) {
    size_t idx = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD + threadIdx.x * ELEMENTS_PER_THREAD;
    
    float prefix = 1.0f;
    if (blockIdx.x > 0) {
        prefix = block_products[blockIdx.x - 1];
    }
    
    if (idx + 3 < n) {
        float4 vals = *reinterpret_cast<float4*>(&output[idx]);
        vals.x *= prefix;
        vals.y *= prefix;
        vals.z *= prefix;
        vals.w *= prefix;
        *reinterpret_cast<float4*>(&output[idx]) = vals;
    } else {
        for (size_t i = 0; i < ELEMENTS_PER_THREAD && idx + i < n; ++i) {
            output[idx + i] *= prefix;
        }
    }
}

/**
 * Host function to compute the cumulative product of the input vector.
 * Allocates memory, launches kernels, and cleans up.
 */
extern "C" void solution(const float* input, float* output, size_t n) {
    // Calculate number of blocks
    size_t num_blocks = (n + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    
    // Allocate device memory for block products
    float* block_products;
    cudaMalloc(&block_products, num_blocks * sizeof(float));
    
    // Set up grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, 1);
    dim3 gridDim(num_blocks, 1);
    
    // Launch kernel to compute intra-block cumulative products
    cumProdBlockKernel<<<gridDim, blockDim>>>(input, output, block_products, n);
    
    // Compute cumulative products across blocks
    computeCumBlockProducts<<<1, 1>>>(block_products, num_blocks);
    
    // Propagate block products to get final cumulative product
    cumProdPropagateKernel<<<gridDim, blockDim>>>(output, block_products, n);
    
    // Free allocated memory
    cudaFree(block_products);
}