#include <cuda_runtime.h>
#include <math.h>
#include <cub/cub.cuh>

template <int UNROLL_FACTOR>
__global__ __launch_bounds__(256) void layer_norm_4d_kernel_vectorized(
    const float* __restrict__ X, const float* __restrict__ gamma, const float* __restrict__ beta,
    float* Y, size_t B, size_t F, size_t D1, size_t D2, float epsilon
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= B) return;

    const int tid = threadIdx.x;
    constexpr size_t vec_factor = 4;
    const size_t norm_size = F*D1*D2;
    const size_t batch_offset = batch_idx*norm_size;
    const float4* X4 = reinterpret_cast<const float4*>(X + batch_offset);
    const float4* g4 = reinterpret_cast<const float4*> (gamma);
    const float4* b4 = reinterpret_cast<const float4*> (beta);
    float4* Y4 = reinterpret_cast<float4*>(Y + batch_offset);
    size_t vec_sz = norm_size/vec_factor;

    float local_sum = 0.f;
    float local_sum_squares = 0.f;
    #pragma unroll UNROLL_FACTOR
    for (size_t i = tid; i < vec_sz; i += 256) {
        float4 v = __ldg(&X4[i]);
        local_sum += v.x + v.y + v.z + v.w;
        local_sum_squares += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    struct Pair { float s, ss; };
    struct SumOp { __device__ Pair operator()(Pair a, Pair b) const { return {a.s + b.s, a.ss + b.ss}; } };
    __shared__ typename cub::BlockReduce<Pair, 256>::TempStorage tmp;
    Pair agg = cub::BlockReduce<Pair, 256>(tmp).Reduce({local_sum,local_sum_squares}, SumOp());

    __shared__ float mean, inv_std;
    if (tid==0) {
        float norm_size_inv = 1.0f / norm_size;
        mean    =  agg.s * norm_size_inv;
        float var = __fmaf_rn(agg.ss, norm_size_inv, - mean*mean);
        inv_std = rsqrtf(var + epsilon);
    }
    __syncthreads();

    #pragma unroll UNROLL_FACTOR
    for (size_t i = tid; i < vec_sz; i += 256) {
        float4 v = __ldg(&X4[i]);
        float4 G = __ldg(&g4[i]), Bv = __ldg(&b4[i]), N;
        N.x = __fmaf_rn((v.x-mean)*inv_std, G.x, Bv.x);
        N.y = __fmaf_rn((v.y-mean)*inv_std, G.y, Bv.y);
        N.z = __fmaf_rn((v.z-mean)*inv_std, G.z, Bv.z);
        N.w = __fmaf_rn((v.w-mean)*inv_std, G.w, Bv.w);
        Y4[i] = N;
    }
}

extern "C" void solution(const float* X, const float* gamma, const float* beta, float* Y, size_t B, size_t F, size_t D1, size_t D2) {
    size_t norm_size = F*D1*D2;
    int unroll_factor;
    
    if (norm_size > 256*256) unroll_factor = 8;      
    else if (norm_size > 128*128) unroll_factor = 4;  
    else unroll_factor = 2;                     
    
    const float epsilon = 1e-5f;
    
    if (unroll_factor == 8)
        layer_norm_4d_kernel_vectorized<8><<<B, 256>>>(X, gamma, beta, Y, B, F, D1, D2, epsilon);
    else if (unroll_factor == 4)
        layer_norm_4d_kernel_vectorized<4><<<B, 256>>>(X, gamma, beta, Y, B, F, D1, D2, epsilon);
    else
        layer_norm_4d_kernel_vectorized<2><<<B, 256>>>(X, gamma, beta, Y, B, F, D1, D2, epsilon);
}