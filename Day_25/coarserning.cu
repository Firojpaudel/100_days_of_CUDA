#include <cuda_runtime.h>
#include <iostream>
#include <string>
using namespace std;

#define NUM_BINS 26
#define COARSENING_FACTOR 4  // Each thread processes 4 elements

// CUDA kernel using Contiguous Partitioning for Coarsening
__global__ void histo_coarsening_contiguous(char* data, unsigned int length, unsigned int* histo) {
    __shared__ unsigned int histo_s[NUM_BINS];

    // Initialize shared memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes COARSENING_FACTOR consecutive elements
    for (unsigned int i = tid * COARSENING_FACTOR; i < length; i += blockDim.x * gridDim.x * COARSENING_FACTOR) {
        for (int j = 0; j < COARSENING_FACTOR; j++) {
            if (i + j < length) {
                int alphabet_position = data[i + j] - 'a';
                if (alphabet_position >= 0 && alphabet_position < 26) {
                    atomicAdd(&(histo_s[alphabet_position]), 1);
                }
            }
        }
    }
    __syncthreads();

    // Merge to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        atomicAdd(&(histo[bin]), histo_s[bin]);
    }
}

// CUDA kernel using Interleaved Partitioning for Coarsening
__global__ void histo_coarsening_interleaved(char* data, unsigned int length, unsigned int* histo) {
    __shared__ unsigned int histo_s[NUM_BINS];

    // Initialize shared memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histo_s[bin] = 0u;
    }
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes COARSENING_FACTOR elements spaced apart
    for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < COARSENING_FACTOR; j++) {
            unsigned int index = i + j * (blockDim.x * gridDim.x);
            if (index < length) {
                int alphabet_position = data[index] - 'a';
                if (alphabet_position >= 0 && alphabet_position < 26) {
                    atomicAdd(&(histo_s[alphabet_position]), 1);
                }
            }
        }
    }
    __syncthreads();

    // Merge to global memory
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        atomicAdd(&(histo[bin]), histo_s[bin]);
    }
}

// Main function
int main() {
    cout << "Enter a string (lowercase letters only): ";
    string input;
    getline(cin, input);
    unsigned int length = input.length();

    // Allocate and copy data to device
    char* d_data;
    cudaMalloc((void**)&d_data, length * sizeof(char));
    cudaMemcpy(d_data, input.c_str(), length * sizeof(char), cudaMemcpyHostToDevice);

    // Allocate histogram on device
    unsigned int* d_histo_contiguous;
    unsigned int* d_histo_interleaved;
    cudaMalloc((void**)&d_histo_contiguous, NUM_BINS * sizeof(unsigned int));
    cudaMalloc((void**)&d_histo_interleaved, NUM_BINS * sizeof(unsigned int));
    cudaMemset(d_histo_contiguous, 0, NUM_BINS * sizeof(unsigned int));
    cudaMemset(d_histo_interleaved, 0, NUM_BINS * sizeof(unsigned int));

    int blockSize = 256;
    int gridSize = (length + blockSize * COARSENING_FACTOR - 1) / (blockSize * COARSENING_FACTOR);

    // Timing events
    cudaEvent_t start_contiguous, stop_contiguous;
    cudaEvent_t start_interleaved, stop_interleaved;
    cudaEventCreate(&start_contiguous);
    cudaEventCreate(&stop_contiguous);
    cudaEventCreate(&start_interleaved);
    cudaEventCreate(&stop_interleaved);

    // Launch the Contiguous Partitioning kernel
    cudaEventRecord(start_contiguous);
    histo_coarsening_contiguous<<<gridSize, blockSize>>>(d_data, length, d_histo_contiguous);
    cudaEventRecord(stop_contiguous);
    cudaEventSynchronize(stop_contiguous);
    float milliseconds_contiguous = 0;
    cudaEventElapsedTime(&milliseconds_contiguous, start_contiguous, stop_contiguous);

    // Launch the Interleaved Partitioning kernel
    cudaEventRecord(start_interleaved);
    histo_coarsening_interleaved<<<gridSize, blockSize>>>(d_data, length, d_histo_interleaved);
    cudaEventRecord(stop_interleaved);
    cudaEventSynchronize(stop_interleaved);
    float milliseconds_interleaved = 0;
    cudaEventElapsedTime(&milliseconds_interleaved, start_interleaved, stop_interleaved);

    // Copy back results for both kernels
    unsigned int h_histo_contiguous[NUM_BINS];
    unsigned int h_histo_interleaved[NUM_BINS];
    cudaMemcpy(h_histo_contiguous, d_histo_contiguous, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histo_interleaved, d_histo_interleaved, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Display results
    cout << "\nCharacter Frequency Histogram (Contiguous Partitioning):\n";
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_histo_contiguous[i] > 0) {
            cout << char('a' + i) << ": " << h_histo_contiguous[i] << endl;
        }
    }
    cout << "\nKernel Execution Time (Contiguous Partitioning): " << milliseconds_contiguous << " ms\n";

    cout << "\nCharacter Frequency Histogram (Interleaved Partitioning):\n";
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_histo_interleaved[i] > 0) {
            cout << char('a' + i) << ": " << h_histo_interleaved[i] << endl;
        }
    }
    cout << "\nKernel Execution Time (Interleaved Partitioning): " << milliseconds_interleaved << " ms\n";

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_histo_contiguous);
    cudaFree(d_histo_interleaved);
    cudaEventDestroy(start_contiguous);
    cudaEventDestroy(stop_contiguous);
    cudaEventDestroy(start_interleaved);
    cudaEventDestroy(stop_interleaved);

    return 0;
}