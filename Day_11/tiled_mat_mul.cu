#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define THRESHOLD 256

__global__ void tiledMatrixMulKernel(float *M, float *N, float *P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1)/TILE_WIDTH; ++ph) {
        int mIdx = Row * Width + ph * TILE_WIDTH + tx;
        int nIdx = (ph * TILE_WIDTH + ty) * Width + Col;
        
        Mds[ty][tx] = (Row < Width && ph*TILE_WIDTH + tx < Width) ? M[mIdx] : 0;
        Nds[ty][tx] = (Col < Width && ph*TILE_WIDTH + ty < Width) ? N[nIdx] : 0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }

    if (Row < Width && Col < Width) //Added to put boundary check!
        P[Row * Width + Col] = Pvalue;
}

void cpuMatMul(float *A, float *B, float *C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

int main() {
    std::cout << "_________________________________________\n"
              << " GPU/CPU Matrix Multiplication Analyzer \n"
              << "__________________________________________\n";

    int N;
    std::cout << "Enter matrix dimension: ";
    std::cin >> N;

    if(N <= 0) {
        std::cerr << "Error: Matrix size must be positive!\n";
        return 1;
    }

    // Memory allocation
    size_t bytes = N * N * sizeof(float);
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *cpuRes = new float[N*N]{};
    float *gpuRes = new float[N*N]{};

    // Initialize with floating-point values
    std::cout << "\nInitializing matrices (0.0-1.0 range)...\n";
    for(int i = 0; i < N*N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Benchmarking
    double cpuTime = 0, gpuTime = 0;
    bool useGPU = N > THRESHOLD;

    // CPU Execution
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuMatMul(A, B, cpuRes, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration<double>(cpuEnd - cpuStart).count();

    if(useGPU) {
        // GPU Execution with proper timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        float *dM, *dN, *dP;
        cudaMalloc(&dM, bytes);
        cudaMalloc(&dN, bytes);
        cudaMalloc(&dP, bytes);

        cudaEventRecord(start);
        cudaMemcpy(dM, A, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dN, B, bytes, cudaMemcpyHostToDevice);

        dim3 blocks((N + TILE_WIDTH-1)/TILE_WIDTH, (N + TILE_WIDTH-1)/TILE_WIDTH);
        dim3 threads(TILE_WIDTH, TILE_WIDTH);
        tiledMatrixMulKernel<<<blocks, threads>>>(dM, dN, dP, N);

        cudaMemcpy(gpuRes, dP, bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        gpuTime = ms / 1000.0;

        cudaFree(dM); cudaFree(dN); cudaFree(dP);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Results validation
    float maxError = 0.0f;
    if(useGPU) {
        for(int i = 0; i < N*N; ++i) {
            maxError = fmax(maxError, fabs(cpuRes[i] - gpuRes[i]));
        }
    }

    // Display results
    std::cout << "\n---------------- Results ---------------\n";
    std::cout << "Matrix size:        " << N << "x" << N << "\n";
    std::cout << "Compute device:     " << (useGPU ? "GPU" : "CPU") << "\n";
    std::cout << "Threshold:          " << THRESHOLD << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU time:           " << cpuTime << " s\n";
    if(useGPU) {
        std::cout << "GPU time:           " << gpuTime << " s\n";
        std::cout << "Speedup factor:     " << cpuTime/gpuTime << "x\n";
        std::cout << "Maximum error:      " << std::scientific << maxError << "\n";
    }
    std::cout << "--------------------------------------------\n";

    delete[] A; delete[] B; delete[] cpuRes; delete[] gpuRes;
    return 0;
}
