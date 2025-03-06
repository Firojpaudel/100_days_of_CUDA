#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm> // For std::min and std::max

using namespace std;

// Sequential merge function running on the CPU
// Merges two sorted arrays A (size m) and B (size n) into C
void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0; // i for A, j for B, k for C
    // Compare elements from A and B, take the smaller one
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++]; // Copy from A if its element is smaller or equal
        } else {
            C[k++] = B[j++]; // Copy from B if its element is smaller
        }
    }
    // Copy any remaining elements from A
    while (i < m) {
        C[k++] = A[i++];
    }
    // Copy any remaining elements from B
    while (j < n) {
        C[k++] = B[j++];
    }
}

// Device function on GPU to compute co-rank
// Determines how many elements from A should come before position k in C
__device__ int co_rank(int k, int *A, int m, int *B, int n) {
    int i = min(k, m);       // Guess: take k elements from A, capped at m
    int j = k - i;           // Remaining elements from B
    int i_low = max(0, k - n); // Lower bound for i (at least k-n from A)
    
    // Adjust i and j until we find the correct split
    while (true) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            i--; j++; // A’s last element too big, take less from A
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            i++; j--; // B’s last element too big, take more from A
        } else {
            return i; // Correct split found
        }
    }
}

// GPU kernel to merge A and B into C in parallel
__global__ void merge_kernel(int *A, int m, int *B, int n, int *C) {
    // Calculate this block’s portion of the output array C
    int C_curr = blockIdx.x * ((m + n + gridDim.x - 1) / gridDim.x); // Start index
    int C_next = min((blockIdx.x + 1) * ((m + n + gridDim.x - 1) / gridDim.x), m + n); // End index
    int C_length = C_next - C_curr; // Number of elements this block handles

    int thread_id = threadIdx.x; // Thread’s local ID
    int block_size = blockDim.x; // Number of threads in this block

    // Each thread processes elements in strides
    for (int idx = thread_id; idx < C_length; idx += block_size) {
        int k = C_curr + idx; // Global index in C
        int i = co_rank(k, A, m, B, n); // Elements from A up to k
        int j = k - i;        // Elements from B up to k

        // Decide which element (from A or B) goes into C[k]
        if (i < m && (j >= n || A[i] <= B[j])) {
            C[k] = A[i]; // Take from A if it’s smaller or B is exhausted
        } else {
            C[k] = B[j]; // Take from B otherwise
        }
    }
}

// Utility function to print part of an array
void printArray(const char *name, int *arr, int size, int numToPrint = 10) {
    cout << name << ": ";
    for (int i = 0; i < min(size, numToPrint); i++) {
        cout << arr[i] << " ";
    }
    cout << "...\n";
}

int main() {
    // Define sizes of the input arrays
    const int m = 33000; // Size of array A
    const int n = 31000; // Size of array B

    // Print program startup info
    cout << "=== Starting the Merge Program ===\n";
    cout << "Merging two sorted arrays:\n";
    cout << " - Array A size: " << m << " (even numbers)\n";
    cout << " - Array B size: " << n << " (odd numbers)\n";
    cout << " - Output array C size: " << (m + n) << "\n\n";

    // Allocate memory on the host (CPU)
    int *h_A = new int[m];         // Array A: even numbers
    int *h_B = new int[n];         // Array B: odd numbers
    int *h_C_gpu = new int[m + n]; // Merged array from GPU
    int *h_C_cpu = new int[m + n]; // Merged array from CPU for verification

    // Initialize arrays with sorted values
    cout << "Initializing arrays...\n";
    for (int i = 0; i < m; i++) {
        h_A[i] = i * 2; // A = [0, 2, 4, ...]
    }
    for (int i = 0; i < n; i++) {
        h_B[i] = i * 2 + 1; // B = [1, 3, 5, ...]
    }

    // Preview the arrays
    printArray("Array A (first 5)", h_A, m, 5);
    printArray("Array B (first 5)", h_B, n, 5);
    cout << "\n";

    // Allocate memory on the device (GPU)
    cout << "Allocating GPU memory...\n";
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, (m + n) * sizeof(int));
    cout << " - Memory allocated for A, B, and C on GPU\n\n";

    // Copy data from host to device
    cout << "Copying data to GPU...\n";
    cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);
    cout << " - Arrays A and B copied to GPU\n\n";

    // Set up GPU execution parameters
    int blocks = 64;   // Number of blocks
    int threads = 256; // Threads per block
    cout << "GPU launch config:\n";
    cout << " - Blocks: " << blocks << "\n";
    cout << " - Threads per block: " << threads << "\n";
    cout << " - Total threads: " << (blocks * threads) << "\n\n";

    // Set up timing events for GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch GPU kernel and time it
    cout << "Launching GPU merge kernel...\n";
    cudaEventRecord(start);
    merge_kernel<<<blocks, threads>>>(d_A, m, d_B, n, d_C);
    cudaEventRecord(stop);
    cout << " - Kernel launched\n";

    // Copy result back to host
    cout << "Copying merged array back to CPU...\n";
    cudaMemcpy(h_C_gpu, d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);
    cout << " - Result copied from GPU\n";

    // Calculate GPU execution time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "GPU Execution Time: " << milliseconds << " ms\n\n";

    // Run and time the CPU merge
    cout << "Running CPU merge...\n";
    auto cpu_start = chrono::high_resolution_clock::now();
    merge_sequential(h_A, m, h_B, n, h_C_cpu);
    auto cpu_end = chrono::high_resolution_clock::now();
    chrono::duration<float, milli> cpu_duration = cpu_end - cpu_start;
    cout << "CPU Execution Time: " << cpu_duration.count() << " ms\n\n";

    // Verify GPU and CPU results match
    cout << "Verifying GPU vs CPU results...\n";
    bool correct = true;
    for (int i = 0; i < m + n; i++) {
        if (h_C_gpu[i] != h_C_cpu[i]) {
            correct = false;
            cout << " - Mismatch at index " << i << ": GPU = " << h_C_gpu[i] << ", CPU = " << h_C_cpu[i] << "\n";
            break;
        }
    }
    cout << "Result: " << (correct ? "CORRECT" : "INCORRECT") << "\n\n";

    // Show a preview of the merged array
    printArray("Merged Array C (first 10)", h_C_gpu, m + n, 10);

    // Clean up memory
    cout << "Freeing memory...\n";
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_gpu;
    delete[] h_C_cpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cout << " - All memory freed\n";

    return 0;
}