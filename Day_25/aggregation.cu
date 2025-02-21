#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstring> // For strlen
using namespace std;

#define NUM_BINS 26

// CUDA kernel to compute histogram using privatization with aggregation
__global__ void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {
    // Initialize privatized bins
    __shared__ unsigned int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }

    __syncthreads();

    // Histogram with aggregation
    unsigned int accumulator = 0;
    int prevBinIdx = -1;

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
        int alphabet_position = data[i] - 'a';
        
        if (alphabet_position >= 0 && alphabet_position < 26) {
            int bin = alphabet_position;
            
            if (bin == prevBinIdx) {
                ++accumulator;
            } else {
                if (accumulator > 0) {
                    atomicAdd(&(histo_s[prevBinIdx]), accumulator);
                }
                accumulator = 1;
                prevBinIdx = bin;
            }
        }
    }

    if (accumulator > 0) {
        atomicAdd(&(histo_s[prevBinIdx]), accumulator);
    }

    __syncthreads();

    // Commit to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        
        if (binValue > 0) {
            atomicAdd(&(histo[bin]), binValue);
        }
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
    cout << "\nCharacter Frequency Histogram (with aggregation):\n";
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