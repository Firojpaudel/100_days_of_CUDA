#include <cuda_runtime.h>

// Simple softmax kernel that computes softmax along specified dimension
__global__ void softmax_kernel(const float* input, float* output, const size_t* shape, int dim, size_t ndim) {
    // Get global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total size and dimension size
    size_t total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    
    size_t dim_size = shape[dim];
    
    // Calculate strides
    size_t stride_before = 1;
    for (int i = 0; i < dim; i++) {
        stride_before *= shape[i];
    }
    
    size_t stride_after = 1;
    for (int i = dim + 1; i < ndim; i++) {
        stride_after *= shape[i];
    }
    
    // Calculate number of softmax operations (slices)
    size_t num_slices = stride_before * stride_after;
    
    // Return if thread index exceeds slices needed
    if (idx >= num_slices) return;
    
    // Calculate position for this slice
    size_t inner_idx = idx % stride_after;
    size_t outer_idx = idx / stride_after;
    size_t base_idx = (outer_idx * stride_after * dim_size) + inner_idx;
    
    // Find max value for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < dim_size; i++) {
        size_t curr_idx = base_idx + (i * stride_after);
        if (curr_idx < total_size) {
            max_val = fmaxf(max_val, input[curr_idx]);
        }
    }
    
    // Calculate sum of exponentials
    float sum_exp = 0.0f;
    for (int i = 0; i < dim_size; i++) {
        size_t curr_idx = base_idx + (i * stride_after);
        if (curr_idx < total_size) {
            sum_exp += expf(input[curr_idx] - max_val);
        }
    }
    
    // Apply softmax transformation
    for (int i = 0; i < dim_size; i++) {
        size_t curr_idx = base_idx + (i * stride_after);
        if (curr_idx < total_size) {
            output[curr_idx] = expf(input[curr_idx] - max_val) / sum_exp;
        }
    }
}

// Host function to launch kernel
extern "C" void solution(const float* input, int dim, float* output, size_t* shape, size_t ndim) {
    // Copy shape to device - assuming shape is a host pointer
    size_t* d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(size_t));
    cudaMemcpy(d_shape, shape, ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    
    // Calculate strides and number of slices
    size_t stride_before = 1;
    for (size_t i = 0; i < (size_t)dim; i++) {
        stride_before *= shape[i];
    }
    
    size_t stride_after = 1;
    for (size_t i = dim + 1; i < ndim; i++) {
        stride_after *= shape[i];
    }
    
    size_t num_slices = stride_before * stride_after;
    
    // Launch kernel with appropriate grid/block dimensions
    int threadsPerBlock = 512;
    int blocksPerGrid = (num_slices + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, d_shape, dim, ndim);
    
    cudaDeviceSynchronize();
    
    // Free device memory
    cudaFree(d_shape);
}