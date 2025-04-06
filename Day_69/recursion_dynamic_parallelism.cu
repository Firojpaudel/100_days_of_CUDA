#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

__global__ void factorial_kernel_dp(int n, volatile unsigned long long* results, int level) {
    if (n <= 1) {
        atomicExch((unsigned long long*)&results[level], 1ULL);
        __threadfence_system();
        printf("Base case: n = %d, level = %d, result = %llu\n", n, level, results[level]);
        return;
    }

    factorial_kernel_dp<<<1, 1>>>(n - 1, results, level + 1);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        results[level] = 0;
        printf("Child launch failed: n = %d, level = %d: %s\n", n, level, cudaGetErrorString(err));
        return;
    }

    __threadfence_system();
    printf("n = %d, level = %d, child_result = %llu\n", n, level, results[level + 1]);
    unsigned long long temp = static_cast<unsigned long long>(n) * results[level + 1];
    atomicExch((unsigned long long*)&results[level], temp);
    __threadfence_system();
    printf("n = %d, level = %d, result = %llu\n", n, level, results[level]);
}

unsigned long long factorialHost(int n) {
    if (n <= 1) return 1;
    return n * factorialHost(n - 1);
}

int main() {
    int n;
    cout << "Enter the number to get factorial of: ";
    cin >> n;

    if (n < 0) {
        cout << "Please enter a non-negative number!" << endl;
        return 1;
    }
    if (n > 20) {
        cout << "Warning: n > 20 may overflow unsigned long long (max safe factorial is 20!)" << endl;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Device: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;

    size_t max_depth;
    cudaDeviceGetLimit(&max_depth, cudaLimitDevRuntimeSyncDepth);
    cout << "Initial max recursion depth: " << max_depth << endl;

    int requested_depth = n + 1;
    cudaError_t err = cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, requested_depth);
    if (err != cudaSuccess) {
        cout << "Failed to set recursion depth to " << requested_depth << ": " << cudaGetErrorString(err) << endl;
        return 1;
    }
    cudaDeviceGetLimit(&max_depth, cudaLimitDevRuntimeSyncDepth);
    cout << "Updated max recursion depth: " << max_depth << endl;

    unsigned long long* d_results;
    cudaMalloc(&d_results, sizeof(unsigned long long) * (n + 1));
    cudaMemset(d_results, 0, sizeof(unsigned long long) * (n + 1));

    auto start_gpu = chrono::high_resolution_clock::now();
    factorial_kernel_dp<<<1, 1>>>(n, d_results, 0);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> gpu_time = end_gpu - start_gpu;

    unsigned long long* h_results = new unsigned long long[n + 1];
    cudaMemcpy(h_results, d_results, sizeof(unsigned long long) * (n + 1), cudaMemcpyDeviceToHost);
    cout << "Debug: Full results array:" << endl;
    for (int i = 0; i <= n; i++) {
        cout << "Level " << i << ": " << h_results[i] << endl;
    }

    unsigned long long h_result = h_results[0];

    auto start_host = chrono::high_resolution_clock::now();
    unsigned long long host_result = factorialHost(n);
    auto end_host = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> host_time = end_host - start_host;

    cout << "GPU Result (Dynamic Parallelism): " << h_result << endl;
    cout << "GPU Time: " << gpu_time.count() << " ms" << endl;
    cout << "Host Result: " << host_result << endl;
    cout << "Host Time: " << host_time.count() << " ms" << endl;

    delete[] h_results;
    cudaFree(d_results);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }

    return 0;
}