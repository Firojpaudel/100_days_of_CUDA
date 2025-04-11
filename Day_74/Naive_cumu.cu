#include <cuda_runtime.h>

__global__ void prefix_sum_kernel(const float* input, float* output, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        float compensation = 0.0f; // Kahan compensation term; to pass the benchmarks test
        for (unsigned int j = 0; j <= i; j++) {
            float y = input[j] - compensation;
            float t = sum + y;
            compensation = (t - sum) - y;
            sum = t;
        }
        output[i] = sum;
    }
}

extern "C" void solution(const float* input, float* output, size_t N) {
    const int threadsPerBlock = 1024;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    prefix_sum_kernel<<<numBlocks, threadsPerBlock>>>(input, output, static_cast<unsigned int>(N));
}