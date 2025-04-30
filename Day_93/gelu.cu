#include <cuda_runtime.h>
#include <math.h>

__global__ void geluKernelMain(
    const float* __restrict__ A,
    float* __restrict__ C,
    size_t M,
    size_t N
) {
    size_t j = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    if (i < M && j + 3 < N) {
        float4 val = *reinterpret_cast<const float4*>(&A[i * N + j]);
        float4 result;

        float x = val.x;
        float x_cubed = x * x * x;
        float arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        result.x = 0.5f * x * (1.0f + __tanhf(arg));

        x = val.y;
        x_cubed = x * x * x;
        arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        result.y = 0.5f * x * (1.0f + __tanhf(arg));

        x = val.z;
        x_cubed = x * x * x;
        arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        result.z = 0.5f * x * (1.0f + __tanhf(arg));

        x = val.w;
        x_cubed = x * x * x;
        arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        result.w = 0.5f * x * (1.0f + __tanhf(arg));

        *reinterpret_cast<float4*>(&C[i * N + j]) = result;
    }
}

__global__ void geluKernelEdge(
    const float* __restrict__ A,
    float* __restrict__ C,
    size_t M,
    size_t N,
    size_t startJ
) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x + startJ;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    if (i < M && j < N) {
        float x = A[i * N + j];
        float x_cubed = x * x * x;
        float arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        C[i * N + j] = 0.5f * x * (1.0f + tanhf(arg));
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

    dim3 threadsPerBlock(256, 2); 
    size_t N_aligned = (N / 4) * 4; 
    if (N_aligned > 0) {
        dim3 blocksPerGrid((N_aligned + threadsPerBlock.x * 4 - 1) / (threadsPerBlock.x * 4),
                           (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
        geluKernelMain<<<blocksPerGrid, threadsPerBlock>>>(A, C, M, N);
    }
    if (N_aligned < N) {
        dim3 threadsPerBlockEdge(128, 2);
        dim3 blocksPerGridEdge((N - N_aligned + threadsPerBlockEdge.x - 1) / threadsPerBlockEdge.x,
                               (M + threadsPerBlockEdge.y - 1) / threadsPerBlockEdge.y);
        geluKernelEdge<<<blocksPerGridEdge, threadsPerBlockEdge>>>(A, C, M, N, N_aligned);
    }
}