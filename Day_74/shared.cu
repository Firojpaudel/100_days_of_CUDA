#include <cuda_runtime.h>

__global__ void prefix_sum_kernel(const float* input, float* output, unsigned int N) {
    extern __shared__ float temp[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int blockSize = blockDim.x;

    if (i < N) {
        float sum = 0.0f;
        float compensation = 0.0f;

        // Process the input in chunks of size blockDim.x
        for (unsigned int chunk = 0; chunk * blockSize <= i; chunk++) {
            // Load a chunk into shared memory
            unsigned int idx = chunk * blockSize + tid;
            temp[tid] = (idx < N) ? input[idx] : 0.0f;
            __syncthreads();

            // Sum elements in this chunk up to index i
            unsigned int end = (chunk * blockSize + blockSize - 1 > i) ? (i % blockSize) : (blockSize - 1);
            for (unsigned int j = 0; j <= end; j++) {
                float y = temp[j] - compensation;
                float t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
            __syncthreads();
        }

        output[i] = sum;
    }
}

extern "C" void solution(const float* input, float* output, size_t N) {
    const int threadsPerBlock = 1024;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    prefix_sum_kernel<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(input, output, static_cast<unsigned int>(N));
}