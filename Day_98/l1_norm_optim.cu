#include <cuda_runtime.h>

#define blockdim 512

__device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void l1_norm(const float* __restrict__ X, float* __restrict__ Y, size_t B, size_t D) {
    extern __shared__ float sdata[];
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int blockSize = blockDim.x;
    float sum = 0.0f;

    for (int d = tid * 4; d < D - 3; d += blockSize * 4) {
        float4 x = reinterpret_cast<const float4*>(X + b * D)[d >> 2];
        sum += fabsf(x.x) + fabsf(x.y) + fabsf(x.z) + fabsf(x.w);
    }
    for (int d = tid + (D & ~3); d < D; d += blockSize) {
        sum += fabsf(X[b * D + d]);
    }
    sum += 1e-10f;

    sum = warp_reduce(sum);
    if (lane == 0) {
        sdata[tid >> 5] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        float val = tid < (blockDim.x >> 5) ? sdata[tid] : 0.0f;
        val = warp_reduce(val);
        if (tid == 0) {
            sdata[0] = val;
        }
    }
    __syncthreads();

    float norm = sdata[0];
    for (int d = tid * 4; d < D - 3; d += blockSize * 4) {
        float4 x = reinterpret_cast<const float4*>(X + b * D)[d >> 2];
        float4 y;
        y.x = x.x / norm;
        y.y = x.y / norm;
        y.z = x.z / norm;
        y.w = x.w / norm;
        reinterpret_cast<float4*>(Y + b * D)[d >> 2] = y;
    }
    for (int d = tid + (D & ~3); d < D; d += blockSize) {
        Y[b * D + d] = X[b * D + d] / norm;
    }
}

extern "C" void solution(const float* input, float* output, size_t b, size_t d) {
    l1_norm<<<b, blockdim, (blockdim / 32 + 1) * sizeof(float)>>>(input, output, b, d);
}