#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

#define BLOCK_DIM 1024

// Defining the kernel for performing sum reduction using shared memory
__global__ void SharedMemorySumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;

    // Loading elements into shared memory
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    __syncthreads();

    // Performing reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    // Writing the final sum to output
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

int main() {
    int size = 2 * BLOCK_DIM * sizeof(float);
    
    // Allocating host memory
    float* h_input = new float[2 * BLOCK_DIM];
    float h_output;

    // Asking for user input
    cout << "Enter the value to initialize the array: ";
    float value;
    cin >> value;

    // Initializing input data
    for (int i = 0; i < 2 * BLOCK_DIM; i++) {
        h_input[i] = value; // Assigning all elements to user input value
    }

    // Allocating device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, sizeof(float));

    // Copying input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Measuring execution time
    auto start = chrono::high_resolution_clock::now();

    // Launching the kernel
    SharedMemorySumReductionKernel<<<1, BLOCK_DIM>>>(d_input, d_output);

    // Copying the result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Measuring execution time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> duration = end - start;

    // Printing the final sum and execution time
    cout << "Final sum: " << h_output << endl;
    cout << "Execution time: " << duration.count() << " ms" << endl;

    // Freeing allocated memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return 0;
}
