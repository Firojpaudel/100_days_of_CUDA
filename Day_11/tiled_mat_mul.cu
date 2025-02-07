#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define TILE_WIDTH 16

__global__ void tiledMatrixMulKernel(float *M, float *N, float *P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < Width && ph * TILE_WIDTH + tx < Width)
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        if (Col < Width && ph * TILE_WIDTH + ty < Width)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    if (Row < Width && Col < Width)
        P[Row * Width + Col] = Pvalue;
}

void matrixMultiply(float *h_M, float *h_N, float *h_P, int Width) {
    int size = Width * Width * sizeof(float);
    float *d_M, *d_N, *d_P;

    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    tiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU computation time: " << gpu_duration.count() << " seconds" << std::endl;
}

void matrixMultiplyCPU(float *h_M, float *h_N, float *h_P, int Width) {
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            float sum = 0;
            for (int k = 0; k < Width; ++k) {
                sum += h_M[i * Width + k] * h_N[k * Width + j];
            }
            h_P[i * Width + j] = sum;
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU computation time: " << cpu_duration.count() << " seconds" << std::endl;
}

void printMatrix(float *matrix, int Width) {
    for (int i = 0; i < Width; ++i) {
        for (int j = 0; j < Width; ++j) {
            std::cout << matrix[i * Width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int Width;
    std::cout << "Enter the width of the matrices: ";
    std::cin >> Width;

    int size = Width * Width * sizeof(float);
    float *h_M = (float *)malloc(size);
    float *h_N = (float *)malloc(size);
    float *h_P = (float *)malloc(size);
    float *h_P_cpu = (float *)malloc(size);

    std::cout << "Enter elements of matrix M:" << std::endl;
    for (int i = 0; i < Width * Width; ++i) {
        std::cin >> h_M[i];
    }

    std::cout << "Enter elements of matrix N:" << std::endl;
    for (int i = 0; i < Width * Width; ++i) {
        std::cin >> h_N[i];
    }

    matrixMultiplyCPU(h_M, h_N, h_P_cpu, Width);
    matrixMultiply(h_M, h_N, h_P, Width);

    std::cout << "Matrix M:" << std::endl;
    printMatrix(h_M, Width);

    std::cout << "Matrix N:" << std::endl;
    printMatrix(h_N, Width);

    std::cout << "Result matrix P (CPU):" << std::endl;
    printMatrix(h_P_cpu, Width);

    std::cout << "Result matrix P (GPU):" << std::endl;
    printMatrix(h_P, Width);

    free(h_M);
    free(h_N);
    free(h_P);
    free(h_P_cpu);

    return 0;
}
