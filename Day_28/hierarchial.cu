#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

#define BLOCK_DIM 1024
#define COARSE_FACTOR 2  // Define coarsening factor

// Defining the kernel for performing coarsened sum reduction using shared memory
__global__ void CoarsenedSumReductionKernel(float* input, float* output, int numElements) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    // Checking boundary condition
    if (i >= numElements) return;

    // Performing coarsened sum reduction
    float sum = (i < numElements) ? input[i] : 0.0f;
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; ++tile) {
        unsigned int idx = i + tile * BLOCK_DIM;
        if (idx < numElements) {
            sum += input[idx];
        }
    }

    // Storing partial sum in shared memory
    input_s[t] = sum;
    __syncthreads();

    // Performing reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }

    // Using atomic add to combine partial sums
    if (t == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    int numArrays;
    cout << "Enter the number of arrays: ";
    cin >> numArrays;

    int numElements = numArrays * COARSE_FACTOR * 2 * BLOCK_DIM;
    int size = numElements * sizeof(float);

    // Allocating host memory
    float* h_input = new float[numElements];
    float h_output = 0.0f;

    // Initializing input data
    for (int i = 0; i < numElements; i++) {
        h_input[i] = 1.0f; // Assigning all elements to 1.0 for easy summing
    }

    // Allocating device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, sizeof(float));

    // Copying input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    // Measuring execution time
    auto start = high_resolution_clock::now();

    // Calculating grid size
    int numBlocks = (numElements + (COARSE_FACTOR * 2 * BLOCK_DIM) - 1) / (COARSE_FACTOR * 2 * BLOCK_DIM);

    // Launching the kernel
    CoarsenedSumReductionKernel<<<numBlocks, BLOCK_DIM>>>(d_input, d_output, numElements);

    // Synchronizing device
    cudaDeviceSynchronize();

    auto stop = high_resolution_clock::now();
    duration<double> elapsed_time = duration_cast<duration<double>>(stop - start);

    // Copying the result back to host
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    // Printing the final sum and execution time
    cout << "Final sum: " << h_output << endl;
    cout << "Execution time: " << fixed << setprecision(6) << elapsed_time.count() << " seconds" << endl;

    // Freeing allocated memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return 0;
}
