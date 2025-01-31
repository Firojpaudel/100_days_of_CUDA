/*Matrix Multiplication Code:*/
#include<iostream>
#include<cuda_runtime.h>

using namespace std;

__global__ void matmul(float *M, float *N, float *P, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < width && col < width){
        float sum = 0;
        for(int i = 0; i < width; i++){
            sum += M[row * width + i] * N[i * width + col];
        }
        P[row * width + col] = sum;
    }
}

int main(){
    int width;
    cout << "Enter the width of the matrix: ";
    cin >> width;

    size_t size = width * width * sizeof(float); // Size_t is an unsigned integer type of at least 16 bit used to represent the size of an object

    // Allocate memory for host variables
    float *h_M = (float *)malloc(size);
    float *h_N = (float *)malloc(size);
    float *h_P = (float *)malloc(size);

    // Input elements of matrix M 
    cout << "Enter elements of matrix M: ";
    for(int i = 0; i < width * width; i++){
        cin >> h_M[i];
    }

    // Input elements of matrix N
    cout << "Enter elements of matrix N: ";
    for(int i = 0; i < width * width; i++){
        cin >> h_N[i];
    }

    // Allocate memory for device variables
    float *d_M, *d_N, *d_P;
    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_P, size);

    // Copy data from host to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);

    //Kernel call
    matmul<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);

    // Copy data from device to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    cout << "Result matrix P: " << endl;
    for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
            cout << h_P[i * width + j] << " ";
        }
        cout << endl;
    }


    free(h_M);
    free(h_N);
    free(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}