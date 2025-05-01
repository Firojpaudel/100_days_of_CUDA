#include <cuda_runtime.h>
#include <float.h>
#include <cooperative_groups.h>
#include <algorithm> // Added for std::min

using namespace cooperative_groups;

// Warp-level reduction for product using shuffle
__device__ __inline__ float warpReduceProd(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val *= __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void prodReduceKernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t before_size,
    size_t dim_size,
    size_t after_size,
    int dim,
    size_t ndim,
    const size_t* __restrict__ shape
) {
    extern __shared__ float sdata[];

    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t output_size = before_size * after_size;

    // Early exit if thread is out of bounds
    if (idx >= output_size) return;

    size_t before_idx = idx / after_size;
    size_t after_idx = idx % after_size;

    float prod_val = 1.0f;

    // Special case: Reducing the last dimension with contiguous memory
    if (dim == ndim - 1 && after_size == 1) {
        size_t start_idx = before_idx * dim_size;
        size_t stride = blockDim.x * gridDim.x;

        // Each thread processes multiple elements with stride
        for (size_t d = idx; d < before_size * dim_size; d += stride) {
            prod_val *= input[d];
        }
    } else {
        // General case: Strided access
        size_t stride = blockDim.x * gridDim.x;
        for (size_t d = idx; d < before_size * dim_size * after_size; d += stride) {
            size_t dim_idx = (d / after_size) % dim_size;
            size_t input_idx = (before_idx * dim_size + dim_idx) * after_size + after_idx;
            prod_val *= input[input_idx];
        }
    }

    // Warp-level reduction
    prod_val = warpReduceProd(prod_val);

    // Store result in shared memory for block-level reduction
    if (warp.thread_rank() == 0) {
        sdata[tid / 32] = prod_val;
    }

    block.sync();

    // Block-level reduction
    if (tid < 32) {
        prod_val = (tid < blockDim.x / 32) ? sdata[tid] : 1.0f;
        prod_val = warpReduceProd(prod_val);
        if (tid == 0) {
            output[blockIdx.x] = prod_val;
        }
    }
}

extern "C" void solution(
    const float* input,
    int dim,
    float* output,
    size_t* shape,
    size_t ndim
) {
    size_t before_size = 1;
    for (int i = 0; i < dim; i++) {
        before_size *= shape[i];
    }

    size_t dim_size = shape[dim];

    size_t after_size = 1;
    for (int i = dim + 1; i < ndim; i++) {
        after_size *= shape[i];
    }

    size_t output_size = before_size * after_size;

    // Optimize block and grid sizes
    const int blockSize = 256;
    const int warpsPerBlock = blockSize / 32;
    const size_t numBlocks = std::min((output_size + warpsPerBlock - 1) / warpsPerBlock, size_t(65535));

    // Allocate shared memory dynamically
    size_t sharedMemSize = warpsPerBlock * sizeof(float);

    prodReduceKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        input, output, before_size, dim_size, after_size, dim, ndim, shape
    );

    // Ensure kernel completion
    cudaDeviceSynchronize();
}