#include <cuda_runtime.h>

// So similar to yesterdays so why not? 

__global__ void upperTriangularMatMul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (i >= N || j >= N || i > j) return; // Skip lower triangle and out-of-bounds

    float sum = 0.0f;
    #pragma unroll 4
    for (int k = i; k <= j; k++) {
        sum += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

extern "C" void solution(const float* A, const float* B, float* C, size_t N) {
    if (N <= 0) return;

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    upperTriangularMatMul<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
}