#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t error = call;                                                \
        if (error != cudaSuccess) {                                              \
            printf("CUDA error %04d: %s file: %s line: %d\n", error,             \
                   cudaGetErrorString(error), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

void printArray(int *arr, int n, const char *label) {
    cout << label << ": ";
    for (int i = 0; i < n; i++) cout << arr[i] << " ";
    cout << endl;
}

// GPU kernel to merge two sorted arrays
__global__ void mergeKernel(int *A, int m, int *B, int n, int *C) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  // My thread ID
    int totalThreads = blockDim.x * gridDim.x;
    
    int elementsPerThread = (m + n + totalThreads - 1) / totalThreads;  // My chunk size
    int k_start = tid * elementsPerThread;  // Where I start
    int k_end = min(k_start + elementsPerThread, m + n);  // Where I stop
    
    if (k_start >= m + n) return;  
    
    // Binary search to find my split point
    int i_low = max(0, k_start - n), i_high = min(k_start, m);
    int i = (i_low + i_high) / 2, j = k_start - i;
    
    while (i_low <= i_high) {
        if (i < m && j > 0 && A[i] < B[j-1]) i_low = i + 1;
        else if (j < n && i > 0 && B[j] < A[i-1]) i_high = i - 1;
        else break;
        i = (i_low + i_high) / 2; j = k_start - i;
    }
    
    // Merge my chunk
    int k = k_start;
    while (k < k_end && i < m && j < n) {
        if (A[i] <= B[j]) C[k++] = A[i++];
        else C[k++] = B[j++];
    }
    while (k < k_end && i < m) C[k++] = A[i++];
    while (k < k_end && j < n) C[k++] = B[j++];
}

int main() {
    const int N = 16;          // Total elements
    const int numSegments = 4; // Initial chunks
    
    int h_input[N] = {52, 95, 13, 42, 72, 15, 13, 19, 6, 86, 38, 89, 60, 54, 72, 7};
    
    cout << "Starting parallel merge sort process..." << endl;  
    printArray(h_input, N, "Original Array");
    
    // Step 1: Sort chunks on CPU
    const int segmentSize = N / numSegments;  // 4 elements each
    for (int i = 0; i < numSegments; ++i) {
        sort(h_input + i * segmentSize, h_input + (i + 1) * segmentSize);
    }
    cout << "After sorting " << numSegments << " initial segments of size " << segmentSize << ":" << endl;
    printArray(h_input, N, "Sorted Segments");
    
    // Step 2: GPU setup
    int *d_inputA, *d_inputB, *d_output;
    CUDA_CHECK(cudaMalloc(&d_inputA, sizeof(int) * N/2));  // Max size needed
    CUDA_CHECK(cudaMalloc(&d_inputB, sizeof(int) * N/2));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int) * N));
    
    cudaEvent_t start, stop;  // For timing
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float totalTime = 0;
    
    // Step 3: Merge on GPU
    for (int step = segmentSize; step < N; step *= 2) {  // 4 -> 8 -> 16
        cout << "\nMerge step with subarray size " << step << ":" << endl;  // Show step size
        int pairs = N / (2 * step);  // How many merges
        for (int pair = 0; pair < pairs; pair++) {
            int i = pair * 2 * step;
            int sizeA = min(step, N - i);
            int sizeB = min(step, N - i - sizeA);
            if (sizeB <= 0) break;
            
            // Copy to GPU
            CUDA_CHECK(cudaMemcpy(d_inputA, h_input + i, sizeof(int) * sizeA, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_inputB, h_input + i + sizeA, sizeof(int) * sizeB, cudaMemcpyHostToDevice));
            
            const int blockSize = 256;
            const int gridSize = (sizeA + sizeB + blockSize - 1) / blockSize;
            
            // Run and time kernel
            CUDA_CHECK(cudaEventRecord(start));
            mergeKernel<<<gridSize, blockSize>>>(d_inputA, sizeA, d_inputB, sizeB, d_output);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            totalTime += milliseconds;
            
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy back
            CUDA_CHECK(cudaMemcpy(h_input + i, d_output, sizeof(int) * (sizeA + sizeB), cudaMemcpyDeviceToHost));
            // Detailed output like you wanted
            cout << "Merged segments " << pair*2 << " and " << pair*2 + 1 << " (pos " << i << "-" 
                 << i + sizeA + sizeB - 1 << "): ";
            printArray(h_input + i, sizeA + sizeB, "Result");
        }
    }
    
    cout << "\nFinal Results:" << endl; 
    printArray(h_input, N, "Fully Sorted Array");
    cout << "Total kernel execution time: " << totalTime << " ms" << endl;
    cout << "Number of merge steps: " << static_cast<int>(log2(static_cast<float>(numSegments))) << endl;
    cout << "Average time per merge step: " << totalTime / log2(static_cast<float>(numSegments)) << " ms" << endl;
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_inputA));
    CUDA_CHECK(cudaFree(d_inputB));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}