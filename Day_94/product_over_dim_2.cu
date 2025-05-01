#include <cuda_runtime.h>
#include <algorithm>

__global__ void prod_reduce_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  size_t M, size_t S_d, size_t N) {
    size_t out_idx = blockIdx.x;
    size_t m = out_idx / N;
    size_t n = out_idx % N; // Use modulo for clarity and potential compiler optimization

    // Calculate base pointer for the input slice
    const float* base = input + (m * S_d * N + n);

    // Initialize product in double precision to avoid floating-point errors
    double prod = 1.0;

    // Strided loop to reduce warp divergence and improve memory coalescing
    for (size_t k = threadIdx.x; k < S_d; k += blockDim.x) {
        prod *= static_cast<double>(base[k * N]);
    }

    // Warp-level reduction using shuffle instructions
    constexpr unsigned FULL_MASK = 0xffffffffu;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        prod *= __shfl_down_sync(FULL_MASK, prod, offset);
    }

    // Shared memory for inter-warp reduction
    __shared__ double warp_prod[32]; // Optimized for max 1024 threads (32 warps)
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Store warp-level results in shared memory
    if (lane == 0) {
        warp_prod[wid] = prod;
    }
    __syncthreads();

    // Final block-level reduction using the first warp
    if (wid == 0) {
        double block_prod = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_prod[lane] : 1.0;
        for (int offset = ((blockDim.x + warpSize - 1) / warpSize) / 2; offset > 0; offset >>= 1) {
            block_prod *= __shfl_down_sync(FULL_MASK, block_prod, offset);
        }
        if (lane == 0) {
            output[out_idx] = static_cast<float>(block_prod);
        }
    }
}


extern "C" void solution(const float* input, int dim, float* output, size_t* shape, size_t ndim) {
    // Copy shape from device to host
    std::vector<size_t> hshape(ndim);
    cudaMemcpy(hshape.data(), shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);

    // Compute M (pre-reduction dims), N (post-reduction dims), and S_d (reduction dim)
    size_t M = 1, N = 1, S_d = hshape[dim];
    for (int i = 0; i < dim; ++i) {
        M *= hshape[i];
    }
    for (int i = dim + 1; i < static_cast<int>(ndim); ++i) {
        N *= hshape[i];
    }

    // Early exit for invalid cases
    size_t total_outputs = M * N;
    if (total_outputs == 0 || S_d == 0) {
        return;
    }

    // Determine optimal block size (multiple of 32, up to 1024)
    int block_size = 1024;
    while (block_size < static_cast<int>(S_d) && block_size <= 1024) {
        block_size <<= 1;
    }
    block_size = std::min(block_size, 1024);

    // Launch kernel
    dim3 grid(total_outputs);
    dim3 block(block_size);
    prod_reduce_kernel<<<grid, block>>>(input, output, M, S_d, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error appropriately (e.g., throw exception or log)
    }
}