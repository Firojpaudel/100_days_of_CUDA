#include <cuda_runtime.h>

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication, bias addition, Swish activation, and scaling
__global__ void matrixMulSwishKernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float scaling_factor,
    float* __restrict__ output,
    size_t batch_size,
    size_t in_features,
    size_t out_features
) {
    // Shared memory for tiles of input and weight matrices
    __shared__ float s_input[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_weight[TILE_WIDTH][TILE_WIDTH];

    // Thread indices within the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices for the output matrix
    int row = blockIdx.y * TILE_WIDTH + ty;  // Row in output (batch dimension)
    int col = blockIdx.x * TILE_WIDTH + tx;  // Column in output (out_features dimension)

    float sum = 0.0f;

    // Tiled matrix multiplication
    for (int k = 0; k < in_features; k += TILE_WIDTH) {
        // Load tile from input matrix
        if (row < batch_size && (k + tx) < in_features) {
            s_input[ty][tx] = input[row * in_features + (k + tx)];
        } else {
            s_input[ty][tx] = 0.0f;  // Padding with zeros if out of bounds
        }

        // Load tile from weight matrix (implicitly handling transpose)
        if (col < out_features && (k + ty) < in_features) {
            s_weight[tx][ty] = weight[col * in_features + (k + ty)]; // Access as weightᵀ
        } else {
            s_weight[tx][ty] = 0.0f;  // Padding with zeros if out of bounds
        }
        __syncthreads();  // Synchronize to ensure tiles are loaded

        // Compute partial dot product
        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += s_input[ty][i] * s_weight[tx][i];
        }
        __syncthreads();  // Synchronize before loading next tile
    }

    // Compute final output for this thread
    if (row < batch_size && col < out_features) {
        // Add bias
        sum += bias[col];

        // Apply Swish activation: z * sigmoid(z)
        float sigmoid = 1.0f / (1.0f + __expf(-sum));
        float swish = sum * sigmoid;

        // Apply scaling factor
        output[row * out_features + col] = scaling_factor * swish;
    }
}

// Host function to launch the kernel
extern "C" void solution(
    const float* input_matrix,
    const float* weight_matrix,
    const float* bias,
    float scaling_factor,
    float* output,
    size_t batch_size,
    size_t in_features,
    size_t out_features
) {
    // Handle edge case
    if (batch_size == 0 || in_features == 0 || out_features == 0) {
        cudaMemset(output, 0, sizeof(float) * batch_size * out_features);
        return;
    }

    // Configure grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(
        (out_features + TILE_WIDTH - 1) / TILE_WIDTH,
        (batch_size + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Launch the kernel
    matrixMulSwishKernel<<<blocksPerGrid, threadsPerBlock>>>(
        input_matrix, weight_matrix, bias, scaling_factor, output,
        batch_size, in_features, out_features
    );
}