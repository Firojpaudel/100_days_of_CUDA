#include<cudnn.h>
#include<cuda_runtime.h>

#include<iostream>

using namespace std;

// Error Checking Macro - for cuDNN
#define CHECK_CUDNN(call) {                                                                             \
    cudnnStatus_t status = call;                                                                        \
    if (status != CUDNN_STATUS_SUCCESS) {                                                               \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << " (line " << __LINE__ << ")\n";  \
        exit(1);                                                                                        \
    }                                                                                                   \
}

// CUDA check macro
#define CHECK_CUDA(call) {                                                                              \
    cudaError_t err = call;                                                                             \
    if (err != cudaSuccess) {                                                                           \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (line " << __LINE__ << ")\n";       \
        exit(1);                                                                                        \
    }                                                                                                   \
}

int main() {
    // Create cuDNN handle for library operations
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Define input tensor: 4 samples (batch size), 2 channels, 1x1 spatial dims
    cudnnTensorDescriptor_t inputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, 
                                           CUDNN_TENSOR_NCHW,      // NCHW format: batch, channels, height, width
                                           CUDNN_DATA_FLOAT,       // Data type: 32-bit float
                                           4, 2, 1, 1));           // Shape: 4x2x1x1

    // Sample input data: 4 samples with 2 features each
    float h_input[8] = {
        1.0, 2.0,  // Sample 1: feature 0, feature 1
        3.0, 4.0,  // Sample 2
        5.0, 6.0,  // Sample 3
        7.0, 8.0   // Sample 4
    };

    // Allocate and initialize device memory for input
    float *d_input;
    CHECK_CUDA(cudaMalloc(&d_input, 8 * sizeof(float)));                     // Allocate 8 floats on GPU
    CHECK_CUDA(cudaMemcpy(d_input, h_input, 8 * sizeof(float), cudaMemcpyHostToDevice)); // Copy input to GPU

    // Define batch norm parameters: scale and bias per channel (2 channels)
    cudnnTensorDescriptor_t bnParamDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bnParamDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bnParamDesc, 
                                           CUDNN_TENSOR_NCHW, 
                                           CUDNN_DATA_FLOAT, 
                                           1, 2, 1, 1));           // Shape: 1x2x1x1 for per-channel params

    // Initialize batch norm parameters on host
    float h_scale[2] = {1.0, 1.0};  // Gamma (scale) for each channel
    float h_bias[2] = {0.0, 0.0};   // Beta (bias) for each channel
    float *d_scale, *d_bias, *d_mean, *d_variance;
    CHECK_CUDA(cudaMalloc(&d_scale, 2 * sizeof(float)));                    // Allocate scale on GPU
    CHECK_CUDA(cudaMalloc(&d_bias, 2 * sizeof(float)));                     // Allocate bias on GPU
    CHECK_CUDA(cudaMalloc(&d_mean, 2 * sizeof(float)));                     // Allocate mean (computed during BN)
    CHECK_CUDA(cudaMalloc(&d_variance, 2 * sizeof(float)));                 // Allocate variance (computed during BN)
    CHECK_CUDA(cudaMemcpy(d_scale, h_scale, 2 * sizeof(float), cudaMemcpyHostToDevice)); // Copy scale to GPU
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, 2 * sizeof(float), cudaMemcpyHostToDevice));   // Copy bias to GPU

    // Allocate device memory for output (same shape as input: 4x2x1x1)
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, 8 * sizeof(float)));

    // Allocate memory for running mean and variance (used for inference)
    float *d_runningMean, *d_runningVariance;
    CHECK_CUDA(cudaMalloc(&d_runningMean, 2 * sizeof(float)));              // Running mean per channel
    CHECK_CUDA(cudaMalloc(&d_runningVariance, 2 * sizeof(float)));          // Running variance per channel
    CHECK_CUDA(cudaMemset(d_runningMean, 0, 2 * sizeof(float)));            // Initialize running mean to 0
    CHECK_CUDA(cudaMemset(d_runningVariance, 0, 2 * sizeof(float)));        // Initialize running variance to 0

    // Parameters for batch normalization
    float alpha = 1.0f, beta = 0.0f;         // Scaling factors for input/output
    double epsilon = 1e-5;                   // Small constant to prevent division by zero
    double expAvgFactor = 0.1;               // Momentum for updating running mean/variance

    // Perform batch normalization in training mode
    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        cudnn,
        CUDNN_BATCHNORM_PER_ACTIVATION,      // Normalize each activation independently
        &alpha, &beta,                       // Input/output scaling factors
        inputDesc, d_input,                  // Input tensor and data
        inputDesc, d_output,                 // Output tensor and data (same shape)
        bnParamDesc, d_scale, d_bias,        // Scale and bias parameters
        expAvgFactor,                        // Momentum for running stats
        d_runningMean, d_runningVariance,    // Updated running mean and variance
        epsilon,                             // Epsilon for numerical stability
        d_mean, d_variance));                // Save mean and variance for backprop

    // Copy normalized output back to host
    float h_output[8];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, 8 * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the batch-normalized output
    cout << "Batch Normalized Output:\n";
    for (int i = 0; i < 4; i++) {
        cout << "Sample " << i << ": " << h_output[i * 2] << ", " << h_output[i * 2 + 1] << "\n";
    }

    // Free GPU memory
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_scale));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_mean));
    CHECK_CUDA(cudaFree(d_variance));
    CHECK_CUDA(cudaFree(d_runningMean));
    CHECK_CUDA(cudaFree(d_runningVariance));

    // Destroy cuDNN descriptors and handle
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bnParamDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}