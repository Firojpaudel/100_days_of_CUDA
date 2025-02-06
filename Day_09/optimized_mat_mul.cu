#include<iostream>
#include<cuda_runtime.h>

#define tile_size 32 // tile size for shared memory

using namespace std;

__global__ void matMulOptim(int *A, int *B, int* C, int Width){
    __shared__ int Asub[tile_size][tile_size]; // shared memory for A
    __shared__ int Bsub[tile_size][tile_size]; // shared memory for B

    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;
    int Pvalue = 0;

    for (int i = 0; i < Width/tile_size; i++){
        // load data from global memory to shared memory
        Asub[threadIdx.y][threadIdx.x] = A[row*Width + (i*tile_size + threadIdx.x)];
        Bsub[threadIdx.y][threadIdx.x] = B[(i*tile_size + threadIdx.y)*Width + col];
        __syncthreads(); // wait for all threads to finish loading


        // calculate the partial sum
        for (int j = 0; j < tile_size; j++){
            Pvalue += Asub[threadIdx.y][j] * Bsub[j][threadIdx.x];
        }
        __syncthreads();
    }

    // Store the final result in C
    if (row < Width && col < Width){
        C[row*Width + col] = Pvalue;
    }
}

int main(){
    const int Width = 1024;
    size_t size = Width * Width * sizeof(int);

    int *h_A = new int[Width * Width];
    int *h_B = new int[Width * Width];
    int *h_C = new int[Width * Width];

    // Initialize the matrices with random values
    for (int i = 0; i < Width * Width; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Allocate memory on the device
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(tile_size, tile_size);
    dim3 blocksPerGrid(Width/ tile_size, Width / tile_size);

    // Launch the kernel
    matMulOptim<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Width);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        cout << "Sample output (top-left 5x5 matrix):\n";
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << h_C[i * Width + j] << " ";
        }
        cout << endl;
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

