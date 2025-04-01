#include<cudnn.h>
#include<cuda_runtime.h>

#include<iostream>

using namespace std;

// Error Checking Macro - for cuDNN
#define CHECK_CUDNN(call){                                                          \
    cudnnStatus_t status = (call);                                                  \
    if (status != CUDNN_STATUS_SUCCESS){                                            \
        fprintf(stderr, "cuDNN Error: %s\n", cudnnGetErrorString(status));          \
        exit(1);                                                                    \
    }                                                                               \
}

// CUDA check macro
#define CHECK_CUDA(call) {                                                          \
    cudaError_t err = (call);                                                       \
    if (err != cudaSuccess) {                                                       \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));               \
        exit(1);                                                                    \
    }                                                                               \
}

int main(){
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Input tensor:
    cudnnTensorDescriptor_t inputDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 4, 2, 1, 1));

    // Sample input:
    float h_input[8]={
        1.0, 2.0, //Sample 1
        3.0, 4.0, //Sample 2
        5.0, 6.0, //Sample 3
        7.0, 8.0  //Sample 4
    };

    float *d_input;

    CHECK_CUDA(cudaMalloc(&d_input, 8 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, 8 * sizeof(float), cudaMemcpyHostToDevice));

    // Batch Norm Params (scaling and bias per channel)
    cudnnTensorDescriptor_t bnParamDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bnParamDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bnParamDesc, 
                                           CUDNN_TENSOR_NCHW, 
                                           CUDNN_DATA_FLOAT, 
                                           1, 2, 1, 1));

    float h_scale[2] = {1.0, 1.0};  // gamma
    float h_bias[2] = {0.0, 0.0};   // beta
    float *d_scale, *d_bias, *d_mean, *d_variance;
    CHECK_CUDA(cudaMalloc(&d_scale, 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mean, 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_variance, 2 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_scale, h_scale, 2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, 2 * sizeof(float), cudaMemcpyHostToDevice));

    // Output tensor (same shape as input)
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, 8 * sizeof(float)));

    // Batch norm during training: compute mean and variance
    float *d_runningMean, *d_runningVariance;
    CHECK_CUDA(cudaMalloc(&d_runningMean, 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_runningVariance, 2 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_runningMean, 0, 2 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_runningVariance, 0, 2 * sizeof(float)));

    float alpha = 1.0f, beta = 0.0f;
    double epsilon = 1e-5;  // Small constant to avoid division by zero
    double expAvgFactor = 0.1;  // Momentum for running mean/variance

    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        cudnn,
        CUDNN_BATCHNORM_PER_ACTIVATION,  // Normalize per activation (not spatial)
        &alpha, &beta,
        inputDesc, d_input,              // Input
        inputDesc, d_output,             // Output
        bnParamDesc, d_scale, d_bias,    // Scale (gamma) and bias (beta)
        expAvgFactor,                    // Momentum
        d_runningMean, d_runningVariance,// Running mean and variance (updated)
        epsilon,                         // Epsilon
        d_mean, d_variance));            // Save mean and variance for backward pass

    // Copy result back to host
    float h_output[8];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, 8 * sizeof(float), cudaMemcpyDeviceToHost));

    // Print normalized output
    printf("Batch Normalized Output:\n");
    for (int i = 0; i < 4; i++) {
        printf("Sample %d: %.3f, %.3f\n", i, h_output[i * 2], h_output[i * 2 + 1]);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_scale));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_mean));
    CHECK_CUDA(cudaFree(d_variance));
    CHECK_CUDA(cudaFree(d_runningMean));
    CHECK_CUDA(cudaFree(d_runningVariance));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bnParamDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;

}