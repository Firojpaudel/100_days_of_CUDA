#include <cuda_runtime.h>

__global__ void mat_vec_multiply(const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output_c, size_t M, size_t K) {
    extern __shared__ float partial_sums[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;

    if (row < M) {
        float sum = 0.0f;
        // Unrolled loop for partial sum
        for (int k = tx; k < K; k += blockDim.x * 4) {
            if (k < K) sum += input_a[row * K + k] * input_b[k];
            if (k + blockDim.x < K) sum += input_a[row * K + k + blockDim.x] * input_b[k + blockDim.x];
            if (k + 2 * blockDim.x < K) sum += input_a[row * K + k + 2 * blockDim.x] * input_b[k + 2 * blockDim.x];
            if (k + 3 * blockDim.x < K) sum += input_a[row * K + k + 3 * blockDim.x] * input_b[k + 3 * blockDim.x];
        }
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) {
            partial_sums[warp_id] = sum;
        }
        __syncthreads();

        // Reduce warp sums in shared memory (first 32 threads only)
        if (tx < 32) {
            float warp_sum = (tx < blockDim.x / 32) ? partial_sums[tx] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
            }
            if (tx == 0) {
                output_c[row] = warp_sum;
            }
        }
    }
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t M, size_t K) {
    const int BLOCKSIZE_X = 128;  
    const int BLOCKSIZE_Y = 1;   
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(1, M);
    size_t shared_mem_size = (BLOCKSIZE_X / 32) * sizeof(float);  
    mat_vec_multiply<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input_a, input_b, output_c, M, K);
    cudaDeviceSynchronize();
}