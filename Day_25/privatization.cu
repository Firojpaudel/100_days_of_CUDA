#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstring> // For strlen
using namespace std;

#define NUM_BINS 26

// CUDA kernel to compute histogram using privatization via shared memory
__global__ void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
    // Declare shared memory for privatization
    __shared__ unsigned int histo_s[NUM_BINS];

    // Initialize shared memory bins
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    // Compute histogram per block using shared memory
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            // Each block maintains a private copy of the histogram in shared memory
            atomicAdd(&(histo_s[alphabet_position]), 1);
        }
    }
    __syncthreads();

    // Merge shared memory bins into global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        atomicAdd(&(histo[bin]), histo_s[bin]);
    }
}

int main() {
    // Prompting user input
    cout << "Enter a string (lowercase letters only): ";
    string input;
    getline(cin, input);
    
    unsigned int length = input.length();

    // Allocating and copying input data to device
    char* d_data;
    cudaMalloc((void**)&d_data, length * sizeof(char));
    cudaMemcpy(d_data, input.c_str(), length * sizeof(char), cudaMemcpyHostToDevice);

    // Allocating histogram array on device
    unsigned int* d_histo;
    cudaMalloc((void**)&d_histo, NUM_BINS * sizeof(unsigned int));
    cudaMemset(d_histo, 0, NUM_BINS * sizeof(unsigned int));

    // Configuring kernel launch parameters
    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;

    // CUDA Event Timing Variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start kernel execution timer
    cudaEventRecord(start);
    
    // Launch the histogram kernel
    histo_private_kernel<<<gridSize, blockSize>>>(d_data, length, d_histo);

    // Stop kernel execution timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Compute execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copying result back to host
    unsigned int h_histo[NUM_BINS];
    cudaMemcpy(h_histo, d_histo, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Printing histogram results
    cout << "\nCharacter Frequency Histogram:\n";
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_histo[i] > 0) {
            cout << char('a' + i) << ": " << h_histo[i] << endl;
        }
    }

    // Printing execution time
    cout << "\nKernel Execution Time: " << milliseconds << " ms\n";

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_histo);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
