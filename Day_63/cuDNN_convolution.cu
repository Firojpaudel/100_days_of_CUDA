#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

// Error checking macros
#define CHECK_CUDA(call) {                                                                              \
    cudaError_t err = call;                                                                             \
    if (err != cudaSuccess) {                                                                           \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " (line " << __LINE__ << ")\n";       \
        exit(1);                                                                                        \
    }                                                                                                   \
}

#define CHECK_CUDNN(call) {                                                                             \
    cudnnStatus_t status = call;                                                                        \
    if (status != CUDNN_STATUS_SUCCESS) {                                                               \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << " (line " << __LINE__ << ")\n";  \
        exit(1);                                                                                        \
    }                                                                                                   \
}

int main() {
    // Define dimensions
    int N = 1, C = 1, H = 5, W = 5;  // Input: 5x5 image
    int K = 1, R = 3, S = 3;         // Filter: 3x3
    int pad_h = 0, pad_w = 0, u = 1, v = 1;  // No padding, stride 1

    // Compute output size
    int P = (H - R + 2 * pad_h) / u + 1;  // 3
    int Q = (W - S + 2 * pad_w) / v + 1;  // 3

    // Initialize cuDNN
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    // Create descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, P, Q));

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, u, v, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Select algorithm
    cudnnConvolutionFwdAlgo_t algo;
#if CUDNN_MAJOR >= 7
    cudnnConvolutionFwdAlgoPerf_t algoPerf[1];
    int returnedAlgoCount;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, filter_desc, conv_desc, output_desc, 1, &returnedAlgoCount, algoPerf));
    if (returnedAlgoCount > 0 && algoPerf[0].status == CUDNN_STATUS_SUCCESS) {
        algo = algoPerf[0].algo;
    } else {
        std::cerr << "No suitable algorithm found\n";
        exit(1);
    }
#else
    // Fallback for cuDNN < 7.0
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc, output_desc,
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &algo));
#endif

    // Allocate workspace
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    // Allocate data
    float *input_data, *filter_data, *output_data;
    CHECK_CUDA(cudaMalloc(&input_data, N * C * H * W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&filter_data, K * C * R * S * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_data, N * K * P * Q * sizeof(float)));

    // Initialize input (5x5, all 1s)
    float h_input[25] = {
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    };
    CHECK_CUDA(cudaMemcpy(input_data, h_input, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize filter (3x3, all 1s)
    float h_filter[9] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
    };
    CHECK_CUDA(cudaMemcpy(filter_data, h_filter, K * C * R * S * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize output
    CHECK_CUDA(cudaMemset(output_data, 0, N * K * P * Q * sizeof(float)));

    // Run convolution
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, input_desc, input_data, filter_desc, filter_data,
                                        conv_desc, algo, workspace, workspace_size, &beta, output_desc, output_data));

    // Copy output to host
    float h_output[9];
    CHECK_CUDA(cudaMemcpy(h_output, output_data, N * K * P * Q * sizeof(float), cudaMemcpyDeviceToHost));

    // Print input matrix
    std::cout << "\nInput Matrix (5x5):\n";
    std::cout << "-------------------------\n";
    for (int i = 0; i < H; i++) {
        std::cout << "| ";
        for (int j = 0; j < W; j++) {
            std::cout << std::fixed << std::setw(4) << std::setprecision(1) << h_input[i * W + j] << " ";
        }
        std::cout << "|\n";
    }
    std::cout << "-------------------------\n";

    // Print kernel
    std::cout << "\nKernel (3x3):\n";
    std::cout << "-------------\n";
    for (int i = 0; i < R; i++) {
        std::cout << "| ";
        for (int j = 0; j < S; j++) {
            std::cout << std::fixed << std::setw(4) << std::setprecision(1) << h_filter[i * S + j] << " ";
        }
        std::cout << "|\n";
    }
    std::cout << "-------------\n";

    // Print output with explanation
    std::cout << "\nOutput Matrix (3x3):\n";
    std::cout << "-------------\n";
    for (int i = 0; i < P; i++) {
        std::cout << "| ";
        for (int j = 0; j < Q; j++) {
            std::cout << std::fixed << std::setw(4) << std::setprecision(1) << h_output[i * Q + j] << " ";
        }
        std::cout << "|\n";
    }
    std::cout << "-------------\n";

    // Explanation of how output is computed (example for output[0][0])
    std::cout << "\nHow Output is Computed (Example for Output[0][0]):\n";
    std::cout << "The kernel slides over the input with stride 1 and no padding.\n";
    std::cout << "For Output[0][0], the kernel covers the top-left 3x3 of the input:\n";
    std::cout << "  Input[0:3][0:3]:\n";
    std::cout << "    | 1.0 1.0 1.0 |\n";
    std::cout << "    | 1.0 1.0 1.0 |\n";
    std::cout << "    | 1.0 1.0 1.0 |\n";
    std::cout << "  Kernel:\n";
    std::cout << "    | 1.0 1.0 1.0 |\n";
    std::cout << "    | 1.0 1.0 1.0 |\n";
    std::cout << "    | 1.0 1.0 1.0 |\n";
    std::cout << "  Element-wise multiply and sum:\n";
    std::cout << "    (1.0*1.0 + 1.0*1.0 + 1.0*1.0) +\n";
    std::cout << "    (1.0*1.0 + 1.0*1.0 + 1.0*1.0) +\n";
    std::cout << "    (1.0*1.0 + 1.0*1.0 + 1.0*1.0) = 9.0\n";
    std::cout << "This process repeats for each output position, sliding the kernel across the input.\n";

    // Clean up
    CHECK_CUDA(cudaFree(input_data));
    CHECK_CUDA(cudaFree(filter_data));
    CHECK_CUDA(cudaFree(output_data));
    if (workspace) CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(handle));

    return 0;
}