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
        result.x = val.x > 0.0f ? val.x + log1p(__expf(-val.x)) : log1p(__expf(val.x));
        result.y = val.y > 0.0f ? val.y + log1p(__expf(-val.y)) : log1p(__expf(val.y));
        result.z = val.z > 0.0f ? val.z + log1p(__expf(-val.z)) : log1p(__expf(val.z));
        result.w = val.w > 0.0f ? val.w + log1p(__expf(-val.w)) : log1p(__expf(val.w));
        

        *reinterpret_cast<float4*>(&C[i * N + j]) = result;
    } else if (i < M) {
        // Handle edge cases (j + 3 >= N)
        if (j + 0 < N) {
            float x = A[i * N + j + 0];
            C[i * N + j + 0] = x > 0.0f ? x + log1p(__expf(-x)) : log1p(__expf(x));
        }
        if (j + 1 < N) {
            float x = A[i * N + j + 1];
            C[i * N + j + 1] = x > 0.0f ? x + log1p(__expf(-x)) : log1p(__expf(x));
        }
        if (j + 2 < N) {
            float x = A[i * N + j + 2];
            C[i * N + j + 2] = x > 0.0f ? x + log1p(__expf(-x)) : log1p(__expf(x));
        }
        if (j + 3 < N) {
            float x = A[i * N + j + 3];
            C[i * N + j + 3] = x > 0.0f ? x + log1p(__expf(-x)) : log1p(__expf(x));
        }
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

    dim3 threadsPerBlock(64, 4);

    dim3 blocksPerGrid((N + threadsPerBlock.x * 4 - 1) / (threadsPerBlock.x * 4),
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    softplusKernel<<<blocksPerGrid, threadsPerBlock>>>(A, C, M, N);
}