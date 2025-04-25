#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float alpha,
    float* __restrict__ output,
    size_t M,
    size_t N
) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x * 4 + threadIdx.x * 4;
    size_t idx = i * N + j;

    if (i < M && j < N) {
        float4 result;
        if (j + 3 < N) {
            // Vectorized load and compute (assumes 16-byte alignment)
            float4 x = *reinterpret_cast<const float4*>(&input[idx]);
            result.x = fmaxf(x.x, alpha * x.x);
            result.y = fmaxf(x.y, alpha * x.y);
            result.z = fmaxf(x.z, alpha * x.z);
            result.w = fmaxf(x.w, alpha * x.w);
            *reinterpret_cast<float4*>(&output[idx]) = result;
        } else {
            // Boundary case with loop to reduce divergence
            float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int k = 0; k < 4 && j + k < N; ++k) {
                x[k] = input[idx + k];
            }
            result.x = fmaxf(x[0], alpha * x[0]);
            result.y = fmaxf(x[1], alpha * x[1]);
            result.z = fmaxf(x[2], alpha * x[2]);
            result.w = fmaxf(x[3], alpha * x[3]);
            for (int k = 0; k < 4 && j + k < N; ++k) {
                output[idx + k] = (&result.x)[k];
            }
        }
    }
}

extern "C" void solution(
    const float* input,
    float alpha,
    float* output,
    size_t M,
    size_t N
) {
    dim3 threadsPerBlock(32, 8); 
    dim3 blocksPerGrid((N + threadsPerBlock.x * 4 - 1) / (threadsPerBlock.x * 4),
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, alpha, output, M, N);
}