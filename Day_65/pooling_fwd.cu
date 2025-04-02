#include<iostream>
#include<cuda_runtime.h>
#include<math.h>
#include<stdlib.h>

// Error Check Macro
#define CHECK_CUDA_ERROR(call) {                                \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

// Sigmoid activation function (the device version)
__device__ float sigmoid(float x){
    return 1.0/ (1.0f + expf(-x));
}

// Kernel pool fwd
__global__ void pooling_fwd_kernel (float *X, float *Y, float *b, int C, int H, int W, int out_H, int out_W){
    int K =2; //Pool filter size

    // Indexes for height, width and feature map
    int w = blockIdx.x * blockDim.x + threadIdx.x; //Output co-ordinate
    int h = blockIdx.y * blockDim.y + threadIdx.y; //Output co-ordinate 
    int m = blockIdx.z * blockDim.z + threadIdx.z;

    // Bounds check 
    if (m < C && h < out_H && w < out_W){
        float sum = 0.0f;

        // Average Computation: 2 x 2 neighborhood 
        for (int p = 0; p < K; p++){
            for (int q = 0; q < K; q++){
                int in_h = h * K + p; //Mapping output coordinate to input coordinate
                int in_w = w * K + q; //Mapping output coordinate to input coordinate
                sum += X[m* H *W + in_h * W+ in_w]; 
            }
        }

        // Averaging the sum we juss got and adding bias into the activation functionn
        float avg = sum / (K *K);
        Y[m * out_H * out_W + h* out_W + w] = sigmoid(avg + b[m]);
    }
}

// Host Function 
void pooling_fwd(float *h_X, float *h_Y, float *h_b, int C, int H, int W){
    int K = 2;
    int out_H = H/K; //Pooling Function logic H' ← H/K 
    int out_W = W/K;

    // Device mem pointers 
    float *d_X, *d_Y, *d_b;

    // Memory Allocation 
    CHECK_CUDA_ERROR(cudaMalloc(&d_X, C*H*W*sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_Y, C* out_H* out_W * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, C* sizeof(float)));

    // Memory Copy
    CHECK_CUDA_ERROR(cudaMemcpy(d_X, h_X, C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, C * sizeof(float), cudaMemcpyHostToDevice));

    // Block and Grid Dimensions:
    dim3 blockDim(16,16,1);
    dim3 gridDim(
        (out_W + blockDim.x -1) / blockDim.x,
        (out_H + blockDim.y -1) / blockDim.y,
        (C + blockDim.z -1) / blockDim.z
    );

    //Kernel Launch
    pooling_fwd_kernel<<<gridDim, blockDim>>>(d_X, d_Y, d_b, C, H, W, out_H, out_W);

    // Checking the kernel for any sort of errors:
    CHECK_CUDA_ERROR(cudaGetLastError());

    //Sync to ensure the kernel completion 
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Result back to Host from Device (GPU)
    CHECK_CUDA_ERROR(cudaMemcpy(h_Y, d_Y, C*out_H*out_W*sizeof(float), cudaMemcpyDeviceToHost));

    // Freeing the GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_X));
    CHECK_CUDA_ERROR(cudaFree(d_Y));
    CHECK_CUDA_ERROR(cudaFree(d_b));

}

// Example Test func..
int main (){
    // I'll provide the one that was in LeNet-5 figure
    int C= 6;
    int H = 28;
    int W = 28;

    int out_H = H/2;
    int out_W = W/2;

    // Allocating the host memory
    float *h_X = (float *)malloc(C* H* W* sizeof(float));
    float *h_Y = (float *)malloc(C* out_H* out_W *sizeof(float));
    float *h_b = (float *)malloc(C * sizeof(float));

    // Initializing input and biases (example values)
    for (int i = 0; i < C * H * W; i++) h_X[i] = (float)(i % 10); // Dummy data
    for (int m = 0; m < C; m++) h_b[m] = 0.1f * m; // Dummy biases
 
    // Running the forward pass on GPU
     pooling_fwd(h_X, h_Y, h_b, C, H, W);
 
    // Printing a sample output (e.g., first feature map, first few values)
     printf("Sample output for feature map 0:\n");
     for (int h = 0; h < 3; h++) {
         for (int w = 0; w < 3; w++) {
             printf("%.4f ", h_Y[h * out_W + w]);
         }
         printf("\n");
     }
 
    // Free host memory.....
     free(h_X);
     free(h_Y);
     free(h_b);
 
     return 0;

}