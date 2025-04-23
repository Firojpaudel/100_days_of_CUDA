#include <cuda_runtime.h>

__global__ void vectorAdd(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    size_t N
) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n_vec4 = N / 4;
    if (i < n_vec4) {
        float4 val_a = reinterpret_cast<const float4*>(A)[i];
        float4 val_b = reinterpret_cast<const float4*>(B)[i];
        float4 result;
        result.x = val_a.x + val_b.x;
        result.y = val_a.y + val_b.y;
        result.z = val_a.z + val_b.z;
        result.w = val_a.w + val_b.w;
        reinterpret_cast<float4*>(C)[i] = result;
    }
    if (i == n_vec4 && (N % 4) != 0) {
        if (N % 4 >= 1) C[N - 1] = __ldg(&A[N - 1]) + __ldg(&B[N - 1]);
        if (N % 4 >= 2) C[N - 2] = __ldg(&A[N - 2]) + __ldg(&B[N - 2]);
        if (N % 4 == 3) C[N - 3] = __ldg(&A[N - 3]) + __ldg(&B[N - 3]);
    }
}

extern "C" void solution(
    const float* d_input1,
    const float* d_input2,
    float* d_output,
    size_t n
) {
    if (n == 0) {
        cudaMemset(d_output, 0, sizeof(float) * n);
        return;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n / 4 + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n);
}