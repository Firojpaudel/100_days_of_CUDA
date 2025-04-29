#include <cuda_runtime.h>

__global__ void softplusKernel(
    const float* __restrict__ A,
    float* __restrict__ C,
    size_t M,
    size_t N
) {
    size_t j = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j + 3 < N) {
        float4 val = *reinterpret_cast<const float4*>(&A[i * N + j]);
        float4 result;
        result.x = log1p(__expf(val.x));
        result.y = log1p(__expf(val.y));
        result.z = log1p(__expf(val.z));
        result.w = log1p(__expf(val.w));
        *reinterpret_cast<float4*>(&C[i * N + j]) = result;
    } else if (i < M) {
        if (j + 0 < N) C[i * N + j + 0] = log1p(__expf(A[i * N + j + 0]));
        if (j + 1 < N) C[i * N + j + 1] = log1p(__expf(A[i * N + j + 1]));
        if (j + 2 < N) C[i * N + j + 2] = log1p(__expf(A[i * N + j + 2]));
        if (j + 3 < N) C[i * N + j + 3] = log1p(__expf(A[i * N + j + 3]));
    }
}

extern "C" void solution(
    const float* A,
    float* C,
    size_t M,
    size_t N
) {
    if (M == 0 || N == 0) {
        cudaMemset(C, 0, sizeof(float) * M * N);
        return;
    }

    dim3 threadsPerBlock(128, 8); 
    dim3 blocksPerGrid((N + threadsPerBlock.x * 4 - 1) / (threadsPerBlock.x * 4),
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    softplusKernel<<<blocksPerGrid, threadsPerBlock>>>(A, C, M, N);
}