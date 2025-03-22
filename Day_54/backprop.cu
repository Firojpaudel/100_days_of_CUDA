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

/** Note: (To myself 🙂‍↕️)
 * @param M Number of output channels (feature maps)
 * @param C Number of input channels
 * @param H_in Height of the input
 * @param W_in Width of the input
 * @param K Kernel size (K x K)
 * @param dE_dY Gradient of the error with respect to the output [M * H_out * W_out]
 * @param W Weights of the convolutional layer [M * C * K * K]
 * @param dE_dX Gradient of the error with respect to the input [C * H_in * W_in] (output)
 **/

// Well anyway let's start coding our back prop kernel

// So this is the kernel for Gradient Respect to Input ( i.e ∂E/∂X )
__global__ void convLayer_backward_x_grad_kernel(int M, int C, int H_in, int W_in, int K, const float *dE_dY, const float *W, float *dE_dX)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H_in * W_in;

    if (index < total)
    {
        // Extract c, h, w from the index (row-major order)
        int c = index / (H_in * W_in);
        int h = (index / W_in) % H_in;
        int w = index % W_in;

        // Compute output dimensions
        int H_out = H_in - K + 1;
        int W_out = W_in - K + 1;

        // Initialize dE_dX[c, h, w] to 0
        float sum = 0.0f;

        // Sum over output channels and kernel positions (Σ_m Σ_p Σ_q)
        for (int m = 0; m < M; m++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    // Compute output indices h' = h-p, w' = w-q
                    int h_out = h - p;
                    int w_out = w - q;
                    // Check if h_out, w_out are within bounds
                    if (h_out >= 0 && h_out < H_out && w_out >= 0 && w_out < W_out)
                    {
                        // Compute linear indices for dE_dY and W
                        int idx_dE_dY = m * H_out * W_out + h_out * W_out + w_out;
                        int idx_W = m * C * K * K + c * K * K + (K - 1 - p) * K + (K - 1 - q);
                        // Accumulate: dE_dY[m, h-p, w-q] * W[m, c, K-1-p, K-1-q]
                        sum += dE_dY[idx_dE_dY] * W[idx_W];
                    }
                }
            }
        }

        // Store the result
        dE_dX[index] = sum;
    }
}

// Now time for host function:
void convLayer_backward_x_grad(int M, int C, int H_in, int W_in, int K, const float *dE_dY, const float *W, float *dE_dX){

    // Output dimensions
    int H_out = H_in - K + 1;
    int W_out = W_in - K + 1;

    // Device memory pointers
    float *d_dE_dY, *d_W, *d_dE_dX;

    //Allocating the device memory 
    CUDA_CHECK(cudaMalloc(&d_dE_dY, M*H_out*W_out* sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, M * C * K * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dE_dX, C * H_in * W_in * sizeof(float)));

    // Initialize d_dE_dX to zero on the device
    CUDA_CHECK(cudaMemset(d_dE_dX, 0, C * H_in * W_in * sizeof(float)));

    //CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_dE_dY, dE_dY, M * H_out * W_out * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, W, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice));

    //kernel Launch
    int threadsPerBlock = 256;
    int total = C * H_in * W_in;
    int blocksPerGrid = (total + threadsPerBlock -1) / threadsPerBlock;
    convLayer_backward_x_grad_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, C, H_in, W_in, K, d_dE_dY, d_W, d_dE_dX);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Back to CPU 
    CUDA_CHECK(cudaMemcpy(dE_dX, d_dE_dX, C* H_in* W_in * sizeof(float), cudaMemcpyDeviceToHost));

    //Memory freeee
    cudaFree(d_dE_dY);
    cudaFree(d_W);
    cudaFree(d_dE_dX);
}

// Adding main function to make this code runn
int main() {
    // Test parameters
    int M = 2;      // Number of output channels
    int C = 1;      // Number of input channels
    int H_in = 5;   // Input height
    int W_in = 5;   // Input width
    int K = 3;      // Kernel size

    int H_out = H_in - K + 1; // 3
    int W_out = W_in - K + 1; // 3

    // Allocate host memory
    float *h_dE_dY = new float[M * H_out * W_out];
    float *h_W = new float[M * C * K * K];
    float *h_dE_dX = new float[C * H_in * W_in];

    // Initialize test data
    for (int i = 0; i < M * H_out * W_out; i++) {
        h_dE_dY[i] = 1.0f; // Set dE_dY to all 1s
    }
    for (int i = 0; i < M * C * K * K; i++) {
        h_W[i] = 1.0f; // Set weights to all 1s
    }
    for (int i = 0; i < C * H_in * W_in; i++) {
        h_dE_dX[i] = 0.0f; // Initialize dE_dX to zero
    }

    // Call the host function
    convLayer_backward_x_grad(M, C, H_in, W_in, K, h_dE_dY, h_W, h_dE_dX);

    // Print results to verify
    cout << "Gradient of Error w.r.t. Input (dE_dX) - Input Channel 0:" << endl;
    cout << "Note: Values show how many times each input pixel contributes to the output" << endl;
    cout << "      across two 3x3 filters (all weights = 1, dE_dY = 1)." << endl;
    cout << "     0    1    2    3    4  (w)" << endl;
    for (int h = 0; h < 5; h++) {
        cout.width(2);
        cout << h << " ";
        for (int w = 0; w < 5; w++) {
            cout.width(4);
            cout << h_dE_dX[h * 5 + w] << " ";
        }
        cout << endl;
    }
    cout << "(h)" << endl;

    // Free host memory
    delete[] h_dE_dY;
    delete[] h_W;
    delete[] h_dE_dX;

    return 0;
}