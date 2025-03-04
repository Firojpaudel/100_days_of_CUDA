#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace std::chrono;

// Optimized parallel merge kernel using binary search
__global__ void merge_parallel(int *A, int m, int *B, int n, int *C) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= m + n) return;

    // Performing binary search to find the correct position
    int left = max(0, tid - n);
    int right = min(m, tid);

    while (left < right) {
        int mid = (left + right) / 2;
        if (A[mid] <= B[tid - mid - 1])
            left = mid + 1;
        else
            right = mid;
    }

    int i = left;
    int j = tid - left;

    if (i < m && (j >= n || A[i] <= B[j])) {
        C[tid] = A[i];
    } else {
        C[tid] = B[j];
    }
}

// Sequential merge function for CPU verification
void merge_sequential(int *A, int m, int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j])
            C[k++] = A[i++];
        else
            C[k++] = B[j++];
    }
    while (i < m)
        C[k++] = A[i++];
    while (j < n)
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

    // Generating sorted arrays A and B
    h_A[0] = rand() % 10;
    for (int i = 1; i < m; i++)
        h_A[i] = h_A[i - 1] + (rand() % 10 + 1);

    h_B[0] = rand() % 10;
    for (int i = 1; i < n; i++)
        h_B[i] = h_B[i - 1] + (rand() % 10 + 1);

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

    // Allocating device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * sizeof(int));
    cudaMalloc((void**)&d_B, n * sizeof(int));
    cudaMalloc((void**)&d_C, (m + n) * sizeof(int));

    cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (m + n + blockSize - 1) / blockSize;
    
    auto start_gpu = high_resolution_clock::now();
    merge_parallel<<<gridSize, blockSize>>>(d_A, m, d_B, n, d_C);
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

    cout << "\nCPU Merged Array: ";
    for (int i = 0; i < m + n; i++) cout << h_C_cpu[i] << " ";
    
    cout << "\nGPU Merged Array: ";
    for (int i = 0; i < m + n; i++) cout << h_C_gpu[i] << " ";
    
    cout << "\nResult: " << (correct ? "Match :) " : "Mismatch :( ") << endl;
    cout << "CPU Time: " << duration_cpu.count() << " microseconds" << endl;
    cout << "GPU Time: " << duration_gpu.count() << " microseconds" << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    return 0;
}
