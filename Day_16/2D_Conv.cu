#include <stdio.h>
#include <cuda_runtime.h>

#define FILTER_SIZE 3
#define FILTER_RADIUS (FILTER_SIZE / 2)

// CPU implementation of 2D Convolution
void conv2D_CPU(float *input, float *output, float *filter, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
                    int img_x = x + i;
                    int img_y = y + j;
                    float value = (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) 
                                  ? input[img_y * width + img_x] 
                                  : 0; // Zero-padding
                    sum += filter[(j + FILTER_RADIUS) * FILTER_SIZE + (i + FILTER_RADIUS)] * value;
                }
            }
            output[y * width + x] = sum;
        }
    }
}

// CUDA Kernel for 2D Convolution (Naive)
__global__ void conv2D(float *input, float *output, float *filter, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;

        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
            for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
                int img_x = x + i;
                int img_y = y + j;
                float value = (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) 
                              ? input[img_y * width + img_x] 
                              : 0; // Zero-padding
                sum += filter[(j + FILTER_RADIUS) * FILTER_SIZE + (i + FILTER_RADIUS)] * value;
            }
        }
        output[y * width + x] = sum;
    }
}

// CUDA Kernel for 2D Convolution (Optimized with Shared Memory)
__global__ void conv2D_shared(float *input, float *output, float *filter, int width, int height) {
    __shared__ float s_input[16 + FILTER_SIZE - 1][16 + FILTER_SIZE - 1]; // Tile size + boundary

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int local_x = threadIdx.x + FILTER_RADIUS;
    int local_y = threadIdx.y + FILTER_RADIUS;

    // Load shared memory with boundary handling
    if (x < width && y < height) {
        s_input[local_y][local_x] = input[y * width + x];
    } else {
        s_input[local_y][local_x] = 0;
    }

    // Load halo cells
    if (threadIdx.x < FILTER_RADIUS) {
        int left_x = x - FILTER_RADIUS;
        s_input[local_y][threadIdx.x] = (left_x >= 0) ? input[y * width + left_x] : 0;
    }
    if (threadIdx.x >= blockDim.x - FILTER_RADIUS) {
        int right_x = x + FILTER_RADIUS;
        s_input[local_y][local_x + FILTER_RADIUS] = (right_x < width) ? input[y * width + right_x] : 0;
    }
    if (threadIdx.y < FILTER_RADIUS) {
        int top_y = y - FILTER_RADIUS;
        s_input[threadIdx.y][local_x] = (top_y >= 0) ? input[top_y * width + x] : 0;
    }
    if (threadIdx.y >= blockDim.y - FILTER_RADIUS) {
        int bottom_y = y + FILTER_RADIUS;
        s_input[local_y + FILTER_RADIUS][local_x] = (bottom_y < height) ? input[bottom_y * width + x] : 0;
    }
    __syncthreads();

    // Perform convolution
    if (x < width && y < height) {
        float sum = 0.0f;
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
            for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
                sum += filter[(j + FILTER_RADIUS) * FILTER_SIZE + (i + FILTER_RADIUS)] * 
                       s_input[local_y + j][local_x + i];
            }
        }
        output[y * width + x] = sum;
    }
}

// Function to run convolution on GPU
void run2DConvolution_GPU(float *h_input, float *h_output, float *h_filter, int width, int height, bool optimized) {
    float *d_input, *d_output, *d_filter;
    int size = width * height * sizeof(float);
    int filter_size = FILTER_SIZE * FILTER_SIZE * sizeof(float);

    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);
    cudaMalloc((void **)&d_filter, filter_size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (optimized) {
        conv2D_shared<<<gridSize, blockSize>>>(d_input, d_output, d_filter, width, height);
    } else {
        conv2D<<<gridSize, blockSize>>>(d_input, d_output, d_filter, width, height);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    printf("%s GPU Execution Time: %0.2f ms\n", optimized ? "Optimized" : "Naive", milliseconds);
}

int main() {
    const int width = 5, height = 5;
    float h_input[] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };
    float h_filter[] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };

    float h_output_GPU_naive[width * height], h_output_GPU_optimized[width * height], h_output_CPU[width * height];

    // Run CPU convolution
    conv2D_CPU(h_input, h_output_CPU, h_filter, width, height);

    // Run and time GPU convolution (Naive)
    run2DConvolution_GPU(h_input, h_output_GPU_naive, h_filter, width, height, false);

    // Run and time GPU convolution (Optimized)
    run2DConvolution_GPU(h_input, h_output_GPU_optimized, h_filter, width, height, true);

    // Compare results and check for matches
    bool match_naive = true, match_optimized = true;
    for (int i = 0; i < width * height; i++) {
        if (h_output_CPU[i] != h_output_GPU_naive[i]) {
            match_naive = false;
        }
        if (h_output_CPU[i] != h_output_GPU_optimized[i]) {
            match_optimized = false;
        }
    }

    // Print results in matrix form
    printf("Output:\n");
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%0.1f ", h_output_CPU[y * width + x]);
        }
        printf("\n");
    }

    // Print match results
    printf("\nCPU and GPU Naive match: %s\n", match_naive ? "Yes" : "No");
    printf("CPU and GPU Optimized match: %s\n", match_optimized ? "Yes" : "No");

    return 0;
}
