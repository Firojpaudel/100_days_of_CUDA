#include <stdio.h>
#include <cuda_runtime.h>

#define FILTER_RADIUS 2  // Radius of filter
#define FILTER_SIZE (2 * FILTER_RADIUS + 1)  // Full filter size

// CUDA Kernel for 1D Convolution (Naive)
__global__ void conv1D(float *input, float *output, float *filter, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0f;

        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
            int index = i + j;
            float value = (index >= 0 && index < n) ? input[index] : 0; // Zero-padding
            sum += filter[j + FILTER_RADIUS] * value;
        }

        output[i] = sum;
    }
}

// CUDA Kernel for 1D Convolution (Optimized with Shared Memory)
__global__ void conv1D_shared(float *input, float *output, float *filter, int n) {
    extern __shared__ float s_input[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x + FILTER_RADIUS;

    if (i < n) {
        s_input[local_i] = input[i];
    }
    if (threadIdx.x < FILTER_RADIUS) {
        s_input[threadIdx.x] = (i >= FILTER_RADIUS) ? input[i - FILTER_RADIUS] : 0;
        s_input[local_i + blockDim.x] = (i + blockDim.x < n) ? input[i + blockDim.x] : 0;
    }
    __syncthreads();

    if (i < n) {
        float sum = 0.0f;
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
            sum += filter[j + FILTER_RADIUS] * s_input[local_i + j];
        }
        output[i] = sum;
    }
}

// CPU implementation of 1D Convolution
void conv1D_CPU(float *input, float *output, float *filter, int n) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;

        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
            int index = i + j;
            float value = (index >= 0 && index < n) ? input[index] : 0; // Zero-padding
            sum += filter[j + FILTER_RADIUS] * value;
        }

        output[i] = sum;
    }
}

// Function to run convolution and measure GPU time
void run1DConvolution_GPU(float *h_input, float *h_output, float *h_filter, int n, bool optimized) {
    float *d_input, *d_output, *d_filter;
    cudaMalloc((void **)&d_input, n * sizeof(float));
    cudaMalloc((void **)&d_output, n * sizeof(float));
    cudaMalloc((void **)&d_filter, FILTER_SIZE * sizeof(float));

    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (optimized) {
        conv1D_shared<<<numBlocks, blockSize, (blockSize + 2 * FILTER_RADIUS) * sizeof(float)>>>(d_input, d_output, d_filter, n);
    } else {
        conv1D<<<numBlocks, blockSize>>>(d_input, d_output, d_filter, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);

    printf("%s GPU Execution Time: %0.2f ms\n", optimized ? "Optimized" : "Naive", milliseconds);
}

int main() {
    const int n = 7;
    float h_input[] = {8, 2, 5, 4, 1, 7, 3}; 
    float h_filter[] = {1, 3, 5, 3, 1};
    float h_output_GPU_naive[n], h_output_GPU_optimized[n], h_output_CPU[n];

    // Run and time CPU convolution
    conv1D_CPU(h_input, h_output_CPU, h_filter, n);

    // Run and time GPU convolution (Naive)
    run1DConvolution_GPU(h_input, h_output_GPU_naive, h_filter, n, false);

    // Run and time GPU convolution (Optimized)
    run1DConvolution_GPU(h_input, h_output_GPU_optimized, h_filter, n, true);

    // Compare results
    bool match_naive = true, match_optimized = true;
    for (int i = 0; i < n; i++) {
        if (abs(h_output_CPU[i] - h_output_GPU_naive[i]) > 1e-5) {
            match_naive = false;
        }
        if (abs(h_output_CPU[i] - h_output_GPU_optimized[i]) > 1e-5) {
            match_optimized = false;
        }
    }

    if (match_naive) {
        printf("Naive GPU results match CPU results!\n");
    } else {
        printf("Naive GPU results do not match CPU results!\n");
    }

    if (match_optimized) {
        printf("Optimized GPU results match CPU results!\n");
    } else {
        printf("Optimized GPU results do not match CPU results!\n");
    }

    // Print final output arrays
    printf("Outputs:\n");
    for (int i = 0; i < n; i++) {
        printf("Output[%d] = %0.1f\n", i, h_output_CPU[i]);
    }

    return 0;
}
