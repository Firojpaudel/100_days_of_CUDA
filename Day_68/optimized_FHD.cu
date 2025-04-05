#include <cuda_runtime.h>
#include <iostream>
#include <chrono> // For timing
using namespace std;

// Define struct for k-space data (x, y, z components)
struct KSpaceSample {
    float x, y, z;
};

// Constant memory for k-space chunk (max 64KB, e.g., 5461 samples at 12 bytes each)
__constant__ KSpaceSample k_c[5461];

// Kernel configuration
const int FHD_THREADS_PER_BLOCK = 1024;
const int CHUNK_SIZE = 5461; // Fits in 64KB (5461 * 12 bytes ≈ 65KB, adjust if needed)

// Optimized FHD kernel
__global__ void cmpFhD(float* rFhD, float* iFhD, const float* x, const float* y, const float* z, const float* rMu, const float* iMu, int N, int chunkStart, int chunkSize) {
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x; // Voxel index
    if (n >= N) return;

    // Load voxel coordinates and accumulators into registers
    float xn = x[n];
    float yn = y[n];
    float zn = z[n];
    float rAccum = rFhD[n]; // Read once, accumulate locally
    float iAccum = iFhD[n];

    // Inner loop over k-space chunk
    #pragma unroll 4 // Unroll 4 times (tuned value, adjust experimentally)
    for (int m = 0; m < chunkSize; m++) {
        // Compute phase (phi) using hardware trig
        float phi = k_c[m].x * xn + k_c[m].y * yn + k_c[m].z * zn;
        float cosPhi = __cosf(phi); // Hardware cosine
        float sinPhi = __sinf(phi); // Hardware sine

        // Accumulate contributions
        int mGlobal = chunkStart + m; // Global index in rMu/iMu
        rAccum += rMu[mGlobal] * cosPhi - iMu[mGlobal] * sinPhi;
        iAccum += rMu[mGlobal] * sinPhi + iMu[mGlobal] * cosPhi;
    }

    // Write back to global memory once
    rFhD[n] = rAccum;
    iFhD[n] = iAccum;
}

// Host code to manage kernel launches
void computeFHD(float* d_rFhD, float* d_iFhD, float* d_x, float* d_y, float* d_z, float* d_rMu, float* d_iMu, KSpaceSample* h_k, int M, int N) {
    int numBlocks = (N + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;

    // Process k-space data in chunks
    for (int chunkStart = 0; chunkStart < M; chunkStart += CHUNK_SIZE) {
        int remaining = M - chunkStart;
        int chunkSize = (remaining < CHUNK_SIZE) ? remaining : CHUNK_SIZE;

        // Copy chunk to constant memory
        cudaMemcpyToSymbol(k_c, &h_k[chunkStart], chunkSize * sizeof(KSpaceSample));

        // Launch kernel
        cmpFhD<<<numBlocks, FHD_THREADS_PER_BLOCK>>>(d_rFhD, d_iFhD, d_x, d_y, d_z,
                                                      d_rMu, d_iMu, N, chunkStart, chunkSize);
        cudaDeviceSynchronize(); // Ensure kernel completes before next chunk
    }
}

// Main function with outputs
int main() {
    int M = 1000000; // 1M k-space samples
    int N = 2097152; // 128^3 voxels

    // Allocate and initialize host memory
    KSpaceSample* h_k = new KSpaceSample[M];
    float* h_x = new float[N];
    float* h_y = new float[N];
    float* h_z = new float[N];
    float* h_rMu = new float[M];
    float* h_iMu = new float[M];
    float* h_rFhD = new float[N]();
    float* h_iFhD = new float[N]();

    // Populate input data (simple test values)
    cout << "Initializing input data..." << endl;
    for (int i = 0; i < M; i++) {
        h_k[i] = {static_cast<float>(i) * 0.001f, static_cast<float>(i) * 0.001f, static_cast<float>(i) * 0.001f};
        h_rMu[i] = 1.0f; // Real modulation
        h_iMu[i] = 0.0f; // Imaginary modulation
    }
    for (int i = 0; i < N; i++) {
        h_x[i] = static_cast<float>(i % 128) * 0.01f; // Simple 3D grid
        h_y[i] = static_cast<float>((i / 128) % 128) * 0.01f;
        h_z[i] = static_cast<float>(i / (128 * 128)) * 0.01f;
    }

    // Device memory
    float *d_rFhD, *d_iFhD, *d_x, *d_y, *d_z, *d_rMu, *d_iMu;
    cudaMalloc(&d_rFhD, N * sizeof(float));
    cudaMalloc(&d_iFhD, N * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_rMu, M * sizeof(float));
    cudaMalloc(&d_iMu, M * sizeof(float));

    // Copy data to device
    cout << "Copying data to GPU..." << endl;
    cudaMemcpy(d_rFhD, h_rFhD, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iFhD, h_iFhD, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rMu, h_rMu, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iMu, h_iMu, M * sizeof(float), cudaMemcpyHostToDevice);

    // Time the computation
    cout << "Starting FHD computation..." << endl;
    auto start = chrono::high_resolution_clock::now();
    computeFHD(d_rFhD, d_iFhD, d_x, d_y, d_z, d_rMu, d_iMu, h_k, M, N);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    // Copy results back
    cout << "Copying results back to host..." << endl;
    cudaMemcpy(h_rFhD, d_rFhD, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iFhD, d_iFhD, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Output sample results
    cout << "\nSample FHD Results (first 5 voxels):" << endl;
    for (int i = 0; i < min(5, N); i++) {
        cout << "Voxel " << i << ": rFhD = " << h_rFhD[i] << ", iFhD = " << h_iFhD[i] << endl;
    }

    // Output performance metrics
    cout << "\nPerformance Metrics:" << endl;
    cout << "Execution Time: " << duration.count() << " ms" << endl;
    cout << "Number of Voxels: " << N << endl;
    cout << "Number of K-Space Samples: " << M << endl;
    cout << "Voxels per Second: " << static_cast<double>(N) / (duration.count() / 1000.0) << endl;

    // Cleanup
    cudaFree(d_rFhD); cudaFree(d_iFhD); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_rMu); cudaFree(d_iMu);
    delete[] h_k; delete[] h_x; delete[] h_y; delete[] h_z;
    delete[] h_rMu; delete[] h_iMu; delete[] h_rFhD; delete[] h_iFhD;

    cout << "\nFHD computation complete!" << endl;
    return 0;
}