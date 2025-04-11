#include <cuda_runtime.h>
#include <float.h> 

__global__ void max_pooling_kernel(const float* input, float* output, int H, int kernel_size, int stride, int padding, int dilation, int Hout) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < Hout) {
        int start_idx = stride * i - padding;
        float max_val = -FLT_MAX;  

        for (int m = 0; m < kernel_size; m++) {
            int idx = start_idx + dilation * m;
            float val = (idx >= 0 && idx < H) ? input[idx] : -FLT_MAX;
            max_val = fmaxf(max_val, val);
        }

        output[i] = max_val;
    }
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, int dilation, float* output, size_t H) {
  
    int Hout = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    const int threadsPerBlock = 256;
    const int numBlocks = (Hout + threadsPerBlock - 1) / threadsPerBlock;

    max_pooling_kernel<<<numBlocks, threadsPerBlock>>>(input, output, static_cast<int>(H), kernel_size, stride, padding, dilation, Hout);

    cudaDeviceSynchronize();
}