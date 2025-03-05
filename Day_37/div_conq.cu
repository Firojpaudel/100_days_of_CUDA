#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <algorithm>  // for std::sort

using namespace std;
using namespace std::chrono;

// Merge Path Kernel (Divide and Conquer Parallel Merge)
// Each thread computes its "diagonal" in the merge matrix and merges its portion.
__global__ void merge_path_kernel(const int *A, int m, const int *B, int n, int *C, int totalThreads) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= totalThreads) return;  // Only use the threads we launched

    int total = m + n;
    // Compute how many elements each thread should handle (approximately)
    int elemsPerThread = (total + totalThreads - 1) / totalThreads;

    // Determine the start and end positions (the "diagonals") in the merge matrix for this thread
    int diagStart = tid * elemsPerThread;
    int diagEnd = min(diagStart + elemsPerThread, total);

    // --- Find the starting coordinates (iStart, jStart) for diagStart ---
    int low = max(0, diagStart - n);
    int high = min(diagStart, m);
    int iStart, jStart;
    while(low < high) {
        int mid = (low + high) / 2;
        int j = diagStart - mid;
        if(mid < m && j > 0 && A[mid] < B[j - 1])
            low = mid + 1;
        else
            high = mid;
    }
    iStart = low;
    jStart = diagStart - low;

    // --- Find the ending coordinates (iEnd, jEnd) for diagEnd ---
    low = max(0, diagEnd - n);
    high = min(diagEnd, m);
    int iEnd, jEnd;
    while(low < high) {
        int mid = (low + high) / 2;
        int j = diagEnd - mid;
        if(mid < m && j > 0 && A[mid] < B[j - 1])
            low = mid + 1;
        else
            high = mid;
    }
    iEnd = low;
    jEnd = diagEnd - low;

    // --- Merge the two segments [iStart, iEnd) from A and [jStart, jEnd) from B ---
    int i = iStart, j = jStart, k = diagStart;
    while(i < iEnd && j < jEnd) {
        if(A[i] <= B[j])
            C[k++] = A[i++];
        else
            C[k++] = B[j++];
    }
    while(i < iEnd) {
        C[k++] = A[i++];
    }
    while(j < jEnd) {
        C[k++] = B[j++];
    }
}

// CPU Sequential Merge for Verification
void merge_sequential(const int *A, int m, const int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while(i < m && j < n) {
        if(A[i] <= B[j])
            C[k++] = A[i++];
        else
            C[k++] = B[j++];
    }
    while(i < m)
        C[k++] = A[i++];
    while(j < n)
        C[k++] = B[j++];
}

int main() {
    srand(static_cast<unsigned int>(time(0)));

    int m, n;
    cout << "Enter size of array A: ";
    cin >> m;
    cout << "Enter size of array B: ";
    cin >> n;

    int *h_A = new int[m];
    int *h_B = new int[n];
    int *h_C_cpu = new int[m + n];
    int *h_C_gpu = new int[m + n];

    // Generate truly random numbers and then sort them
    for (int i = 0; i < m; i++) {
        h_A[i] = rand() % 100;  // Random numbers between 0 and 99
    }
    for (int i = 0; i < n; i++) {
        h_B[i] = rand() % 100;
    }
    sort(h_A, h_A + m);
    sort(h_B, h_B + n);

    // Display Input Arrays
    cout << "Array A: ";
    for (int i = 0; i < m; i++) cout << h_A[i] << " ";
    cout << endl;
    cout << "Array B: ";
    for (int i = 0; i < n; i++) cout << h_B[i] << " ";
    cout << endl;

    // CPU merge and timing
    auto start_cpu = high_resolution_clock::now();
    merge_sequential(h_A, m, h_B, n, h_C_cpu);
    auto stop_cpu = high_resolution_clock::now();
    auto duration_cpu = duration_cast<microseconds>(stop_cpu - start_cpu);

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, (m + n) * sizeof(int));

    cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Determine kernel launch configuration:
    int totalElements = m + n;
    int blockSize = 256;
    // We'll launch enough threads so that each thread handles a contiguous block of the output.
    int totalThreads = (totalElements + blockSize - 1) / blockSize * blockSize;
    int gridSize = totalThreads / blockSize;

    auto start_gpu = high_resolution_clock::now();
    merge_path_kernel<<<gridSize, blockSize>>>(d_A, m, d_B, n, d_C, totalThreads);
    cudaDeviceSynchronize();
    auto stop_gpu = high_resolution_clock::now();
    auto duration_gpu = duration_cast<microseconds>(stop_gpu - start_gpu);

    cudaMemcpy(h_C_gpu, d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < m + n; i++) {
        if (h_C_cpu[i] != h_C_gpu[i]) {
            correct = false;
            break;
        }
    }

    // Display merged arrays and timing
    cout << "\nCPU Merged Array: ";
    for (int i = 0; i < m + n; i++) cout << h_C_cpu[i] << " ";
    cout << "\nGPU Merged Array: ";
    for (int i = 0; i < m + n; i++) cout << h_C_gpu[i] << " ";
    cout << "\nResult: " << (correct ? "Match :)" : "Mismatch :(") << endl;
    cout << "CPU Time: " << duration_cpu.count() << " microseconds" << endl;
    cout << "GPU Time: " << duration_gpu.count() << " microseconds" << endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    return 0;
}
