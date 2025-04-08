#include <cuda_runtime.h>

__global__ void l1_norm_sum_kernel(const float* X, float* S, size_t B, size_t D) {
    extern __shared__ float sdata[];
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    float sum = 0.0f;
    for (int d = tid; d < D; d += blockSize) {
        sum += fabsf(X[b * D + d]);
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) S[b] = sdata[0] + 1e-10f;
}

__global__ void l1_norm_div_kernel(const float* X, const float* S, float* Y, size_t B, size_t D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < B && j < D) {
        int idx = i * D + j;
        Y[idx] = X[idx] / S[i];
    }
}

extern "C" void solution(const float* input, float* output, size_t b, size_t d) {
    float* d_sums;
    cudaMalloc(&d_sums, b * sizeof(float));
    const int BLOCK_SIZE_SUM = 256;
    dim3 blockDimSum(BLOCK_SIZE_SUM);
    dim3 gridDimSum(1, b);
    size_t sharedMemSize = BLOCK_SIZE_SUM * sizeof(float);
    l1_norm_sum_kernel<<<gridDimSum, blockDimSum, sharedMemSize>>>(input, d_sums, b, d);
    const int BLOCK_SIZE_X = 128, BLOCK_SIZE_Y = 2;
    dim3 blockDimDiv(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDimDiv((d + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (b + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    l1_norm_div_kernel<<<gridDimDiv, blockDimDiv>>>(input, d_sums, output, b, d);
    cudaFree(d_sums);
}