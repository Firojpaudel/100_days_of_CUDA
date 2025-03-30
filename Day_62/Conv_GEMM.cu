#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

using namespace std;

// Handy macro to check for CUDA errors - I added this to catch any mistakes early
#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        cudaError_t error = call;                                                                   \
        if (error != cudaSuccess) {                                                                 \
            cerr << "CUDA error: " << cudaGetErrorString(error) << " at line " << __LINE__ << endl; \
            exit(EXIT_FAILURE);                                                                     \
        }                                                                                           \
    } while(0)

// Kernel to unroll the input - I'm transforming the input into a matrix here
__global__ void unrollKernel(int C, int H, int W, int K, float* X, float* X_unroll) {
    // I'm assigning each thread to handle one element in the unrolled matrix
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;  // Output height after convolution
    int W_out = W - K + 1;  // Output width after convolution
    int W_unroll = H_out * W_out;  // Total number of output positions (columns in my unrolled matrix)

    if (t < (C * W_unroll)) {  // Making sure I don't go out of bounds
        // Figuring out which channel and output position this thread is working on
        int c = t / W_unroll;  // Which channel I'm in
        int w_unroll = t % W_unroll;  // Column index in my unrolled matrix
        int h_out = w_unroll / W_out;  // Row position in the output
        int w_out = w_unroll % W_out;  // Column position in the output

        // Starting row index for this channel in the unrolled matrix
        int w_base = c * K * K;

        // Now I'm unrolling the KxK patch for this output position
        int idx = 0;  // Counter for the patch elements
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                // Mapping back to the input coordinates
                int h = h_out + p;
                int w = w_out + q;
                // Calculating where this element goes in the unrolled matrix
                int unroll_idx = w_base + idx;
                // Getting the corresponding input index
                int input_idx = c * H * W + h * W + w;
                // Placing the input value into my unrolled matrix
                X_unroll[unroll_idx * W_unroll + w_unroll] = X[input_idx];
                idx++;  // Moving to the next element in the patch
            }
        }
    }
}

// Kernel for matrix multiplication - This is where I compute the convolution as GEMM
__global__ void matrixMultiply(float* A, float* B, float* C, int m, int n, int k) {
    // Each thread gets one element in the output matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row in the output
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column in the output

    if (row < m && col < n) {  // Checking bounds so I don't mess up
        float sum = 0.0f;
        // Doing the dot product between my filter row and unrolled input column
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        // Storing the result in the output
        C[row * n + col] = sum;
    }
}

// My main convolution function using GEMM
void convLayerGemm(int C, int H, int W, int K, int F, float* X, float* filters, float* output) {
    // Calculating the output size - I need this for everything that follows
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;  // Number of columns in my unrolled matrix
    int filter_size = C * K * K;   // Size of each filter when flattened

    // I want to see how much my input expands - this is the expansion ratio
    float expansion_ratio = (float)(K * K * H_out * W_out) / (H * W);
    cout << "Expansion Ratio: " << expansion_ratio << endl;  // Printing it so I know

    // Setting up memory on the GPU - I need space for input, unrolled input, filters, and output
    float *d_X, *d_X_unroll, *d_filters, *d_output;
    CUDA_CHECK(cudaMalloc(&d_X, C * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X_unroll, C * K * K * W_unroll * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_filters, F * filter_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, F * W_unroll * sizeof(float)));

    // Copying my data to the GPU - Gotta get it there to process
    CUDA_CHECK(cudaMemcpy(d_X, X, C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_filters, filters, F * filter_size * sizeof(float), cudaMemcpyHostToDevice));

    // Launching my unroll kernel - This transforms my input
    int threadsPerBlock = 256;  // Chose 256 threads per block, seems reasonable
    int blocks = (C * W_unroll + threadsPerBlock - 1) / threadsPerBlock;  // Enough blocks to cover all elements
    unrollKernel<<<blocks, threadsPerBlock>>>(C, H, W, K, d_X, d_X_unroll);
    CUDA_CHECK(cudaDeviceSynchronize());  // Waiting for it to finish

    // Now launching the matrix multiplication - This does the actual convolution
    dim3 threads(16, 16);  // 2D thread block for rows and columns
    dim3 grid((W_unroll + threads.x - 1) / threads.x, (F + threads.y - 1) / threads.y);  // Grid to cover output size
    matrixMultiply<<<grid, threads>>>(d_filters, d_X_unroll, d_output, F, W_unroll, filter_size);
    CUDA_CHECK(cudaDeviceSynchronize());  // Making sure it's done

    // Getting the result back to the CPU - I want to see my output
    CUDA_CHECK(cudaMemcpy(output, d_output, F * W_unroll * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleaning up - Don’t want memory leaks on the GPU
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_X_unroll));
    CUDA_CHECK(cudaFree(d_filters));
    CUDA_CHECK(cudaFree(d_output));
}

// Testing it all out
int main() {
    // Setting up a small test case - I want to see if this works
    int C = 1;    // Just one channel to keep it simple
    int H = 4;    // 4x4 input image
    int W = 4;
    int K = 3;    // 3x3 kernel
    int F = 1;    // One filter for now
    
    int H_out = H - K + 1;  // Output will be 2x2
    int W_out = W - K + 1;

    // My test input - a 4x4 image with some pattern
    float X[16] = {
        1, 1, 1, 1,
        1, 2, 2, 1,
        1, 2, 2, 1,
        1, 1, 1, 1
    };

    // My filter - a simple edge detector kind of thing
    float filters[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    // Space for the output
    float* output = new float[F * H_out * W_out];

    // Running my convolution
    convLayerGemm(C, H, W, K, F, X, filters, output);

    // Printing the input so I can compare
    cout << "Input:" << endl;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            cout << X[i * W + j] << " ";
        }
        cout << endl;
    }

    // Printing the output to see what I got
    cout << "\nOutput:" << endl;
    for (int i = 0; i < H_out; i++) {
        for (int j = 0; j < W_out; j++) {
            cout << output[i * W_out + j] << " ";
        }
        cout << endl;
    }

    // Freeing my output memory
    delete[] output;
    return 0;
}