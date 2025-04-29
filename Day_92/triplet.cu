#include <cuda_runtime.h>

#define THREADS_PER_TRIPLET 512
#define CHUNK_SIZE 4096

__global__ void tripletLossKernel(
    const float* __restrict__ a,    // B x E
    const float* __restrict__ p,    // B x E
    const float* __restrict__ n,    // B x E
    float* __restrict__ output,     // Scalar
    float margin, int B, int E
) {
    extern __shared__ float sdata[];
    float* s_ap = sdata;                // CHUNK_SIZE
    float* s_an = &sdata[CHUNK_SIZE];   // CHUNK_SIZE
    float* s_sum_ap = &sdata[2 * CHUNK_SIZE]; // THREADS_PER_TRIPLET
    float* s_sum_an = &sdata[2 * CHUNK_SIZE + THREADS_PER_TRIPLET]; // THREADS_PER_TRIPLET

    int i = blockIdx.x;                 // Triplet index
    int tid = threadIdx.x;              // Thread within triplet
    float sum_ap = 0.0f;
    float sum_an = 0.0f;

    if (i < B) {
        // Process E in chunks
        for (int j0 = 0; j0 < E; j0 += CHUNK_SIZE) {
            // Load chunk into shared memory using float4
            #pragma unroll
            for (int j = tid * 4; j < CHUNK_SIZE && (j0 + j) < E; j += THREADS_PER_TRIPLET * 4) {
                int idx = j0 + j;
                float4 a_vec = reinterpret_cast<const float4*>(&a[i * E + idx])[0];
                float4 p_vec = reinterpret_cast<const float4*>(&p[i * E + idx])[0];
                float4 n_vec = reinterpret_cast<const float4*>(&n[i * E + idx])[0];
                s_ap[j + 0] = a_vec.x - p_vec.x;
                s_ap[j + 1] = a_vec.y - p_vec.y;
                s_ap[j + 2] = a_vec.z - p_vec.z;
                s_ap[j + 3] = a_vec.w - p_vec.w;
                s_an[j + 0] = a_vec.x - n_vec.x;
                s_an[j + 1] = a_vec.y - n_vec.y;
                s_an[j + 2] = a_vec.z - n_vec.z;
                s_an[j + 3] = a_vec.w - n_vec.w;
            }
            __syncthreads();

            // Compute partial squared sums
            float partial_ap = 0.0f;
            float partial_an = 0.0f;
            #pragma unroll
            for (int j = tid; j < CHUNK_SIZE && (j0 + j) < E; j += THREADS_PER_TRIPLET) {
                partial_ap += s_ap[j] * s_ap[j];
                partial_an += s_an[j] * s_an[j];
            }
            s_sum_ap[tid] = partial_ap;
            s_sum_an[tid] = partial_an;
            __syncthreads();

            // Reduce partial sums within block
            for (int s = THREADS_PER_TRIPLET / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    s_sum_ap[tid] += s_sum_ap[tid + s];
                    s_sum_an[tid] += s_sum_an[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                sum_ap += s_sum_ap[0];
                sum_an += s_sum_an[0];
            }
            __syncthreads();
        }

        // Compute loss
        if (tid == 0) {
            float d_ap = sqrtf(sum_ap);
            float d_an = sqrtf(sum_an);
            float loss = fmaxf(0.0f, d_ap - d_an + margin);
            // Atomic add to global output
            atomicAdd(output, loss / (float)B);
        }
    }
}

extern "C" void solution(
    const float* a,
    const float* p,
    const float* n,
    float* output,
    float margin,
    size_t B,
    size_t E
) {
    if (B == 0 || E == 0) return;

    // Initialize output to 0
    cudaMemset(output, 0, sizeof(float));

    // Launch triplet loss kernel
    int threads = THREADS_PER_TRIPLET;
    int blocks = B;
    size_t shared_mem = (2 * CHUNK_SIZE + 2 * THREADS_PER_TRIPLET) * sizeof(float);
    tripletLossKernel<<<blocks, threads, shared_mem>>>(a, p, n, output, margin, (int)B, (int)E);

    // Copy result to host
    cudaMemcpy(output, output, sizeof(float), cudaMemcpyDeviceToHost);
}