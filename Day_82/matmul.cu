#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32  // Tile size for C (32x32)
#define VEC_SIZE 2    // half2 loads 2 elements

__global__ void matrixMulKernel(
    const half* __restrict__ input_a, // Matrix A: M x K
    const half* __restrict__ input_b, // Matrix B: K x N
    half* __restrict__ output_c,      // Matrix C: M x N
    size_t m, size_t n, size_t k
) {
    // Shared memory for tiles of A and B
    __shared__ half tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ half tile_b[TILE_SIZE][TILE_SIZE];

    // Global indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // Row of C
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // Column of C

    float sum = 0.0f; // Accumulate in float for precision

    // Iterate over tiles along K dimension
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A with half2
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x * VEC_SIZE;
        if (a_row < m && a_col + VEC_SIZE - 1 < k && threadIdx.x < TILE_SIZE / VEC_SIZE &&
            ((a_row * k + a_col) * sizeof(half) % 4 == 0)) {
            half2 a_vals = *reinterpret_cast<const half2*>(&input_a[a_row * k + a_col]);
            tile_a[threadIdx.y][threadIdx.x * VEC_SIZE + 0] = a_vals.x;
            tile_a[threadIdx.y][threadIdx.x * VEC_SIZE + 1] = a_vals.y;
        } else {
            for (int v = 0; v < VEC_SIZE; v++) {
                int col_idx = t * TILE_SIZE + threadIdx.x * VEC_SIZE + v;
                tile_a[threadIdx.y][threadIdx.x * VEC_SIZE + v] = (a_row < m && col_idx < k) ? 
                    input_a[a_row * k + col_idx] : __float2half(0.0f);
            }
        }

        // Load tile of B with half2
        int b_row = t * TILE_SIZE + threadIdx.y * VEC_SIZE;
        int b_col = col;
        if (b_row + VEC_SIZE - 1 < k && b_col < n && threadIdx.y < TILE_SIZE / VEC_SIZE &&
            ((b_row * n + b_col) * sizeof(half) % 4 == 0)) {
            half2 b_vals = *reinterpret_cast<const half2*>(&input_b[b_row * n + b_col]);
            tile_b[threadIdx.y * VEC_SIZE + 0][threadIdx.x] = b_vals.x;
            tile_b[threadIdx.y * VEC_SIZE + 1][threadIdx.x] = b_vals.y;
        } else {
            for (int v = 0; v < VEC_SIZE; v++) {
                int row_idx = t * TILE_SIZE + threadIdx.y * VEC_SIZE + v;
                tile_b[threadIdx.y * VEC_SIZE + v][threadIdx.x] = (row_idx < k && b_col < n) ? 
                    input_b[row_idx * n + b_col] : __float2half(0.0f);
            }
        }

        __syncthreads();

        // Compute partial sum
        if (row < m && col < n) {
            for (int i = 0; i < TILE_SIZE; i++) {
                sum += __half2float(tile_a[threadIdx.y][i]) * __half2float(tile_b[i][threadIdx.x]);
            }
        }

        __syncthreads();
    }

    // Write result to C
    if (row < m && col < n) {
        output_c[row * n + col] = __float2half(sum);
    }
}

extern "C" void solution(
    const half* input_a,
    const half* input_b,
    half* output_c,
    size_t m, size_t n, size_t k
) {
    // 2D block and grid
    dim3 threadsPerBlock(TILE_SIZE / VEC_SIZE, TILE_SIZE / VEC_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(input_a, input_b, output_c, m, n, k);
}