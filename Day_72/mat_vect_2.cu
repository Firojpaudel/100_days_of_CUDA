#include <cuda_runtime.h>

// Kernel with multiple threads per row
__global__ void mat_vec_multiply(const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output_c, size_t M, size_t K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int k = tx; k < K; k += blockDim.x) {
            sum += input_a[row * K + k] * input_b[k];
        }
        // Atomic add to combine partial sums within block
        atomicAdd(&output_c[row], sum);
    }
}

// Solution function with 2D block/grid
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t M, size_t K) {    
    const int BLOCKSIZE_X = 256;  // Threads along K
    const int BLOCKSIZE_Y = 4;    // Rows per block
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(1, (M + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y);

    mat_vec_multiply<<<blocksPerGrid, threadsPerBlock>>>(input_a, input_b, output_c, M, K);
    cudaDeviceSynchronize();
}