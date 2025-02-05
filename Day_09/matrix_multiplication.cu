#include<iostream>
#include<cuda_runtime.h>

using namespace std;

// Code for matrix multiplication
__global__ void matMul (int *a, int *b, int *c, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int Pvalue = 0;
    if (row < Width && col < Width) {
        for (int k = 0; k < Width; ++k) {
            Pvalue += a[row * Width + k] * b[k * Width + col];
        }
        c[row * Width + col] = Pvalue;
    }
}

int main(){
    const int Width = 3;
    int a[Width][Width] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int b[Width][Width] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    int c[Width][Width] = {0};

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, Width * Width * sizeof(int));
    cudaMalloc((void**)&d_b, Width * Width * sizeof(int));
    cudaMalloc((void**)&d_c, Width * Width * sizeof(int));

    cudaMemcpy(d_a, a, Width * Width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, Width * Width * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(Width, Width);
    dim3 blocksPerGrid(1, 1);

    matMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, Width);

    cudaMemcpy(c, d_c, Width * Width * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < Width; i++) {
        for (int j = 0; j < Width; j++) {
            cout << c[i][j] << " ";
        }
        cout << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}