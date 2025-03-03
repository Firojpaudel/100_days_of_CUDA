#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define N 8  // Number of elements in the array

// CUDA Kernel for Kogge-Stone Prefix Scan
__global__ void koggeStoneScan(int *d_arr, int n) {
    int tid = threadIdx.x;

    // Step-wise prefix sum calculation
    for (int stride = 1; stride < n; stride *= 2) {
        int temp = 0;


        if (tid >= stride) {
            temp = d_arr[tid - stride];
        }
        __syncthreads();

        // Update the current element with the fetched value
        if (tid >= stride) {
            d_arr[tid] += temp;
        }
        __syncthreads();  // Synchronize after updating

        // Print step-by-step results properly (Thread 0 only prints after updates)
        if (tid == 0) {
            printf("Step %d: Stride %d\n", __ffs(stride), stride);  // __ffs(stride) gives step number
            printf("A = [");
            for (int i = 0; i < n; i++) {
                printf("%d%s", d_arr[i], (i == n - 1) ? "]\n\n" : " ");
            }
        }
        __syncthreads();  // Ensure print is done before next stride
    }
}

int main() {
    // Host array (input)
    int h_arr[N] = {4, 6, 7, 1, 2, 8, 5, 2};

    // Device array
    int *d_arr;
    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch Kernel with N threads (since N=8)
    cudaEventRecord(start);
    koggeStoneScan<<<1, N>>>(d_arr, N);
    cudaEventRecord(stop);

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output final result
    cout << "Final Prefix Sum (Kogge-Stone): ";
    for (int i = 0; i < N; i++) {
        cout << h_arr[i] << " ";
    }
    cout << endl;

    // Print execution time
    cout << "Kernel Execution Time: " << milliseconds << " ms" << endl;

    // Cleanup
    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
