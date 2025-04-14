#include <cuda_runtime.h>
#define K_BATCH 4
__global__ void tensorMatrixMultKernel(
    const float* A, const float* B, float* C,
    size_t I_dim, size_t J_dim, size_t L_dim, size_t K_dim
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = I_dim * J_dim * ((K_dim + K_BATCH - 1) / K_BATCH);
    if (idx < total_threads) {
        size_t i = idx / (((K_dim + K_BATCH - 1) / K_BATCH) * J_dim);
        size_t j = (idx / ((K_dim + K_BATCH - 1) / K_BATCH)) % J_dim;
        size_t k_base = (idx % ((K_dim + K_BATCH - 1) / K_BATCH)) * K_BATCH;
        size_t a_base = (i * J_dim + j) * L_dim;
        size_t c_base = (i * J_dim + j) * K_dim + k_base;
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        for (size_t l = 0; l < L_dim - 3; l += 4) {
            float4 a_vals = *reinterpret_cast<const float4*>(&A[a_base + l]);
            size_t b_idx = l * K_dim + k_base;
            if (k_base + 3 < K_dim) {
                float4 b_vals = *reinterpret_cast<const float4*>(&B[b_idx]);
                sum0 += a_vals.x * b_vals.x;
                sum1 += a_vals.x * b_vals.y;
                sum2 += a_vals.x * b_vals.z;
                sum3 += a_vals.x * b_vals.w;
                b_vals = *reinterpret_cast<const float4*>(&B[b_idx + K_dim]);
                sum0 += a_vals.y * b_vals.x;
                sum1 += a_vals.y * b_vals.y;
                sum2 += a_vals.y * b_vals.z;
                sum3 += a_vals.y * b_vals.w;
                b_vals = *reinterpret_cast<const float4*>(&B[b_idx + 2 * K_dim]);
                sum0 += a_vals.z * b_vals.x;
                sum1 += a_vals.z * b_vals.y;
                sum2 += a_vals.z * b_vals.z;
                sum3 += a_vals.z * b_vals.w;
                b_vals = *reinterpret_cast<const float4*>(&B[b_idx + 3 * K_dim]);
                sum0 += a_vals.w * b_vals.x;
                sum1 += a_vals.w * b_vals.y;
                sum2 += a_vals.w * b_vals.z;
                sum3 += a_vals.w * b_vals.w;
            } else {
                if (k_base + 0 < K_dim) {
                    float b_val = B[b_idx];
                    sum0 += a_vals.x * b_val;
                    sum0 += a_vals.y * B[b_idx + K_dim];
                    sum0 += a_vals.z * B[b_idx + 2 * K_dim];
                    sum0 += a_vals.w * B[b_idx + 3 * K_dim];
                }
                if (k_base + 1 < K_dim) {
                    float b_val = B[b_idx + 1];
                    sum1 += a_vals.x * b_val;
                    sum1 += a_vals.y * B[b_idx + 1 + K_dim];
                    sum1 += a_vals.z * B[b_idx + 1 + 2 * K_dim];
                    sum1 += a_vals.w * B[b_idx + 1 + 3 * K_dim];
                }
                if (k_base + 2 < K_dim) {
                    float b_val = B[b_idx + 2];
                    sum2 += a_vals.x * b_val;
                    sum2 += a_vals.y * B[b_idx + 2 + K_dim];
                    sum2 += a_vals.z * B[b_idx + 2 + 2 * K_dim];
                    sum2 += a_vals.w * B[b_idx + 2 + 3 * K_dim];
                }
                if (k_base + 3 < K_dim) {
                    float b_val = B[b_idx + 3];
                    sum3 += a_vals.x * b_val;
                    sum3 += a_vals.y * B[b_idx + 3 + K_dim];
                    sum3 += a_vals.z * B[b_idx + 3 + 2 * K_dim];
                    sum3 += a_vals.w * B[b_idx + 3 + 3 * K_dim];
                }
            }
        }
        for (size_t l = (L_dim / 4) * 4; l < L_dim; ++l) {
            float a_val = A[a_base + l];
            size_t b_idx = l * K_dim + k_base;
            if (k_base + 0 < K_dim) sum0 += a_val * B[b_idx];
            if (k_base + 1 < K_dim) sum1 += a_val * B[b_idx + 1];
            if (k_base + 2 < K_dim) sum2 += a_val * B[b_idx + 2];
            if (k_base + 3 < K_dim) sum3 += a_val * B[b_idx + 3];
        }
        if (k_base + 0 < K_dim) C[c_base + 0] = sum0;
        if (k_base + 1 < K_dim) C[c_base + 1] = sum1;
        if (k_base + 2 < K_dim) C[c_base + 2] = sum2;
        if (k_base + 3 < K_dim) C[c_base + 3] = sum3;
    }
}
extern "C" void solution(
    const float* A, const float* B, float* C,
    size_t i, size_t j, size_t l, size_t k
) {
    size_t total_threads = i * j * ((k + K_BATCH - 1) / K_BATCH);
    int threadsPerBlock = 128; // Further tuned for H100
    int blocksPerGrid = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
    tensorMatrixMultKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, i, j, l, k);
}