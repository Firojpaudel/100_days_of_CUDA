#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Error checking macro
#define CUDA_CHECK(call)                                                                                                    \
    do {                                                                                                                    \
        cudaError_t err = call;                                                                                             \
        if (err != cudaSuccess) {                                                                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(EXIT_FAILURE);                                                                                             \
        }                                                                                                                   \
    } while (0)

// Kernel for Gradient with respect to Input (∂E/∂X)
__global__ void convLayer_backward_x_grad_kernel(int M, int C, int H_in, int W_in, int K, const float *dE_dY, const float *W, float *dE_dX)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H_in * W_in;

    if (index < total)
    {
        int c = index / (H_in * W_in);
        int h = (index / W_in) % H_in;
        int w = index % W_in;

        int H_out = H_in - K + 1;
        int W_out = W_in - K + 1;

        float sum = 0.0f;

        for (int m = 0; m < M; m++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    int h_out = h - p;
                    int w_out = w - q;
                    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out)
                    {
                        int idx_dE_dY = m * H_out * W_out + h_out * W_out + w_out;
                        int idx_W = m * C * K * K + c * K * K + (K - 1 - p) * K + (K - 1 - q);
                        sum += dE_dY[idx_dE_dY] * W[idx_W];
                    }
                }
            }
        }
        dE_dX[index] = sum;
    }
}

// Host function for ∂E/∂X
void convLayer_backward_x_grad(int M, int C, int H_in, int W_in, int K, const float *dE_dY, const float *W, float *dE_dX)
{
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    float *d_dE_dY, *d_W, *d_dE_dX;

    CUDA_CHECK(cudaMalloc(&d_dE_dY, M * H_out * W_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, M * C * K * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dE_dX, C * H_in * W_in * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_dE_dX, 0, C * H_in * W_in * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_dE_dY, dE_dY, M * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int total = C * H_in * W_in;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    convLayer_backward_x_grad_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, C, H_in, W_in, K, d_dE_dY, d_W, d_dE_dX);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dE_dX, d_dE_dX, C * H_in * W_in * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_dE_dY);
    cudaFree(d_W);
    cudaFree(d_dE_dX);
}

// Kernel for Gradient with respect to Weights (∂E/∂W)
__global__ void convLayer_backward_w_grad_kernel(int M, int C, int H_in, int W_in, int K, const float *dE_dY, const float *X, float *dE_dW)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * C * K * K;

    if (index < total)
    {
        int m = index / (C * K * K);
        int c = (index / (K * K)) % C;
        int p = (index / K) % K;
        int q = index % K;

        int H_out = H_in - K + 1;
        int W_out = W_in - K + 1;

        float sum = 0.0f;

        for (int h_out = 0; h_out < H_out; h_out++)
        {
            for (int w_out = 0; w_out < W_out; w_out++)
            {
                int h_in = h_out + p;
                int w_in = w_out + q;
                int idx_dE_dY = m * H_out * W_out + h_out * W_out + w_out;
                int idx_X = c * H_in * W_in + h_in * W_in + w_in;
                sum += dE_dY[idx_dE_dY] * X[idx_X];
            }
        }
        dE_dW[index] = sum;
    }
}

// Host function for ∂E/∂W
void convLayer_backward_w_grad(int M, int C, int H_in, int W_in, int K, const float *dE_dY, const float *X, float *dE_dW)
{
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    float *d_dE_dY, *d_X, *d_dE_dW;

    CUDA_CHECK(cudaMalloc(&d_dE_dY, M * H_out * W_out * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, C * H_in * W_in * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dE_dW, M * C * K * K * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_dE_dW, 0, M * C * K * K * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_dE_dY, dE_dY, M * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X, C * H_in * W_in * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int total = M * C * K * K;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    convLayer_backward_w_grad_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, C, H_in, W_in, K, d_dE_dY, d_X, d_dE_dW);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dE_dW, d_dE_dW, M * C * K * K * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_dE_dY);
    cudaFree(d_X);
    cudaFree(d_dE_dW);
}

// Combined backpropagation function
void convLayer_backward(int M, int C, int H_in, int W_in, int K, const float *dE_dY, const float *X, const float *W, float *dE_dX, float *dE_dW)
{
    convLayer_backward_x_grad(M, C, H_in, W_in, K, dE_dY, W, dE_dX);
    convLayer_backward_w_grad(M, C, H_in, W_in, K, dE_dY, X, dE_dW);
}

// Main function with test
int main()
{
    // Test params
    int M = 2;      // Number of output channels
    int C = 1;      // Number of input channels
    int H_in = 5;   // Input height
    int W_in = 5;   // Input width
    int K = 3;      // Kernel size

    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1; 


    float *h_X = new float[C * H_in * W_in];
    float *h_W = new float[M * C * K * K];
    float *h_dE_dY = new float[M * H_out * W_out];
    float *h_dE_dX = new float[C * H_in * W_in];
    float *h_dE_dW = new float[M * C * K * K];


    for (int i = 0; i < C * H_in * W_in; i++) {
        h_X[i] = 1.0f; 
    }
    for (int i = 0; i < M * C * K * K; i++) {
        h_W[i] = 1.0f; 
    }
    for (int i = 0; i < M * H_out * W_out; i++) {
        h_dE_dY[i] = 1.0f;
    }
    for (int i = 0; i < C * H_in * W_in; i++) {
        h_dE_dX[i] = 0.0f; 
    }
    for (int i = 0; i < M * C * K * K; i++) {
        h_dE_dW[i] = 0.0f; 
    }

    // Calling  the combined backpropagation function
    convLayer_backward(M, C, H_in, W_in, K, h_dE_dY, h_X, h_W, h_dE_dX, h_dE_dW);

    // Cool print!
    cout << "Gradient of Error w.r.t. Input (dE_dX) - Input Channel 0:" << endl;
    cout << "Note: Values show how many times each input pixel contributes to the output" << endl;
    cout << "      across two 3x3 filters (all weights = 1, dE_dY = 1)." << endl;
    cout << "     0    1    2    3    4  (w)" << endl;
    for (int h = 0; h < H_in; h++) {
        cout.width(2);
        cout << h << " ";
        for (int w = 0; w < W_in; w++) {
            cout.width(4);
            cout << h_dE_dX[h * W_in + w] << " ";
        }
        cout << endl;
    }
    cout << "(h)" << endl;

    cout << "\nGradient of Error w.r.t. Weights (dE_dW):" << endl;
    for (int m = 0; m < M; m++) {
        cout << "Output Channel " << m << ", Input Channel 0:" << endl;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int idx = m * C * K * K + 0 * K * K + p * K + q;
                cout.width(4);
                cout << h_dE_dW[idx] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    
    delete[] h_X;
    delete[] h_W;
    delete[] h_dE_dY;
    delete[] h_dE_dX;
    delete[] h_dE_dW;

    return 0;
}