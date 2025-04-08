#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float alpha, float* output, size_t M, size_t N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        int idx = i * N + j;
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : alpha * val;
    }
}

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    const int BLOCK_SIZE_X = 256;
    const int BLOCK_SIZE_Y = 1;
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((m + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (n + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    leaky_relu_kernel<<<gridDim, blockDim>>>(input, alpha, output, n, m);
}