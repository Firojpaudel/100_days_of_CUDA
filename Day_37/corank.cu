#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace chrono;

// Function to find co-rank (partitioning index)
__device__ int co_rank(int k, int* A, int m, int* B, int n) {
    int left = max(0, k - n);
    int right = min(k, m);

    while (left < right) {
        int mid = (left + right) / 2;
        if (mid < m && k - mid > 0 && A[mid] < B[k - mid - 1]) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// Sequential merge function (used per thread for merging subarrays)
__device__ void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] < B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

// CUDA Kernel for parallel merge
__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = ceil((float)(m + n) / (blockDim.x * gridDim.x));

    int k_curr = tid * elementsPerThread;
    int k_next = min((tid + 1) * elementsPerThread, m + n);

    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

int main() {
    int m, n;
    
    // Take input sizes
    cout << "Enter size of first sorted array (A): ";
    cin >> m;
    cout << "Enter size of second sorted array (B): ";
    cin >> n;

    int* h_A = new int[m];
    int* h_B = new int[n];
    int* h_C = new int[m + n];

    // Input elements for A
    cout << "Enter " << m << " sorted elements for array A: ";
    for (int i = 0; i < m; i++) {
        cin >> h_A[i];
    }

    // Input elements for B
    cout << "Enter " << n << " sorted elements for array B: ";
    for (int i = 0; i < n; i++) {
        cin >> h_B[i];
    }

    // Display input arrays
    cout << "\nInput Array A: ";
    for (int i = 0; i < m; i++) {
        cout << h_A[i] << " ";
    }

    cout << "\nInput Array B: ";
    for (int i = 0; i < n; i++) {
        cout << h_B[i] << " ";
    }
    cout << "\n";

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, (m + n) * sizeof(int));

    cudaMemcpy(d_A, h_A, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (m + n + threadsPerBlock - 1) / threadsPerBlock;

    // Measure execution time
    auto start = high_resolution_clock::now();
    
    merge_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, m, d_B, n, d_C);
    cudaDeviceSynchronize();
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    cudaMemcpy(h_C, d_C, (m + n) * sizeof(int), cudaMemcpyDeviceToHost);

    // Display merged array
    cout << "Merged Array: ";
    for (int i = 0; i < m + n; i++) {
        cout << h_C[i] << " ";
    }
    cout << "\nExecution Time: " << duration.count() << " microseconds\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
