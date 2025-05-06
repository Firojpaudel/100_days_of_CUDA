#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>  // for uintptr_t

#define TILE_SIZE 64
#define VEC_SIZE 2

// Optimized Matrix Multiplication Kernel with Fused Multiply-Add (hfma2) and improved memory access
__global__ void matrixMulKernelOpt(
    const half* __restrict__ input_a,
    const half* __restrict__ input_b,
    half* __restrict__ output_c,
    size_t m, size_t n, size_t k
) {
    __shared__ half tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ half tile_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    half2 sum_half2 = __float2half2_rn(0.0f);  // Use half2 for accumulation

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x * VEC_SIZE;

        // Load A tile into shared memory
        if (a_row < m && a_col + 1 < k && (reinterpret_cast<uintptr_t>(&input_a[a_row * k + a_col]) % 4 == 0)) {
            const half2* a_ptr = reinterpret_cast<const half2*>(&input_a[a_row * k + a_col]);
            half2 a_val = *a_ptr;
            tile_a[threadIdx.y][threadIdx.x * VEC_SIZE + 0] = a_val.x;
            tile_a[threadIdx.y][threadIdx.x * VEC_SIZE + 1] = a_val.y;
        } else {
            for (int i = 0; i < VEC_SIZE; ++i) {
                int ac = a_col + i;
                tile_a[threadIdx.y][threadIdx.x * VEC_SIZE + i] =
                    (a_row < m && ac < k) ? input_a[a_row * k + ac] : __float2half(0.0f);
            }
        }

        int b_row = t * TILE_SIZE + threadIdx.y * VEC_SIZE;
        int b_col = col;

        // Load B tile into shared memory
        if (b_col < n && b_row + 1 < k && (reinterpret_cast<uintptr_t>(&input_b[b_row * n + b_col]) % 4 == 0)) {
            const half2* b_ptr = reinterpret_cast<const half2*>(&input_b[b_row * n + b_col]);
            half2 b_val = *b_ptr;
            tile_b[threadIdx.y * VEC_SIZE + 0][threadIdx.x] = b_val.x;
            tile_b[threadIdx.y * VEC_SIZE + 1][threadIdx.x] = b_val.y;
        } else {
            for (int i = 0; i < VEC_SIZE; ++i) {
                int br = b_row + i;
                tile_b[threadIdx.y * VEC_SIZE + i][threadIdx.x] =
                    (br < k && b_col < n) ? input_b[br * n + b_col] : __float2half(0.0f);
            }
        }

        __syncthreads();

        if (row < m && col < n) {
            // Unroll the loop for tile computation
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i) {
                // Convert each element to half2 and perform HFMA
                half2 a_val = __half2half2(tile_a[threadIdx.y][i]);
                half2 b_val = __half2half2(tile_b[i][threadIdx.x]);
                sum_half2 = __hfma2(a_val, b_val, sum_half2);  // HFMA for half2
            }
        }

        __syncthreads();
    }


    if (row < m && col < n) {
        float sum = __half2float(sum_half2.x) + __half2float(sum_half2.y);  // Accumulate the half2 results
        output_c[row * n + col] = __float2half(sum);
    }
}


extern "C" void solution(
    const half* input_a,
    const half* input_b,
    half* output_c,
    size_t m, size_t n, size_t k
) {

    dim3 threadsPerBlock(TILE_SIZE / VEC_SIZE, TILE_SIZE / VEC_SIZE); // Threads per block (16x16 threads)
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);  // Blocks per grid


    matrixMulKernelOpt<<<blocksPerGrid, threadsPerBlock>>>(input_a, input_b, output_c, m, n, k);


    cudaDeviceSynchronize();
}
