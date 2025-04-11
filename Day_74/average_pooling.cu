#include <cuda_runtime.h>

__global__ void average_pooling_kernel(const float* input, float* output, int H, int kernel_size, int stride, int padding, int Hout) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < Hout) {
        int start_idx = stride * i - padding;
        float sum = 0.0f;

        #pragma unroll
        for (int m = 0; m < kernel_size; m++) {
            int idx = start_idx + m;
            float val = (idx >= 0 && idx < H) ? input[idx] : 0.0f;
            sum += val;
        }

        output[i] = sum / static_cast<float>(kernel_size);
    }
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    int Hout = (H + 2 * padding - kernel_size) / stride + 1;
    const int threadsPerBlock = 512;
    const int numBlocks = (Hout + threadsPerBlock - 1) / threadsPerBlock;
    average_pooling_kernel<<<numBlocks, threadsPerBlock>>>(input, output, static_cast<int>(H), kernel_size, stride, padding, Hout);
    cudaDeviceSynchronize();
}

/*Got runtime of  0.13ms and 461GFlops*/