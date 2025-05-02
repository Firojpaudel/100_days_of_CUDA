#include <cuda_runtime.h>

__global__ void conv2D(const float* __restrict__ A, const float* __restrict__ B,
float *C, size_t H, size_t W, size_t Kh, size_t Kw){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < H && col < W){
        int r_h = (Kh-1) / 2;
        int r_w = (Kw-1) / 2;
        float sum = 0.0f;

        for(int k_h = 0; k_h < Kh; ++k_h){
            for(int k_w = 0; k_w < Kw; ++k_w){
                int curr_row = row - r_h + k_h;
                int curr_col = col - r_w + k_w;

                if(curr_row >= 0 && curr_row < H && curr_col >= 0 && curr_col < W){
                    sum  += A[curr_row * W + curr_col] * B[k_h * Kw + k_w];
                }
            }
        }

        C[row * W + col] = sum;
    }
}

extern "C" void solution(const float* A, const float* B, float* C, size_t H, size_t W, size_t Kh, size_t Kw) {    
    dim3 blockDim(32,16);
    dim3 gridDim((W + blockDim.x - 1)/ blockDim.x, (H + blockDim.y -1)/ blockDim.y);

    conv2D<<<gridDim, blockDim>>>(A, B, C, H, W, Kh, Kw);
    cudaDeviceSynchronize();
}