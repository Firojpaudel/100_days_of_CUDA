#include <cuda_runtime.h>

__global__ void mat_vec_multiply(const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output_c, size_t M, size_t K) {
    extern __shared__ float partial_sums[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int k = tx; k < K; k += blockDim.x) {
            sum += input_a[row * K + k] * input_b[k];
        }
        partial_sums[tx] = sum;

        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tx < s) {
                partial_sums[tx] += partial_sums[tx + s];
            }
            __syncthreads();
        }

        if (tx == 0) {
            output_c[row] = partial_sums[0];
        }
    }
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t M, size_t K) {
    const int BLOCKSIZE_X = 128;
    const int BLOCKSIZE_Y = 1;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(1, M);
    size_t shared_mem_size = BLOCKSIZE_X * sizeof(float);
    mat_vec_multiply<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input_a, input_b, output_c, M, K);
    cudaDeviceSynchronize();
}
