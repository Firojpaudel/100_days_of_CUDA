## Summary of Day 63:

> Exploring CUDNN Library

## What is it?
- The library developed by NVIDIA for accelerating Deep Neural Networks based on CUDA. 
- Used in frameworks like Tensorflow, Pytorch etc.
- Requires input/output data to reside in GPU Memory.

> [!important]
> Convolution paramters in CUDNN:
> | Parameters | Meaning| 
> |------------|--------|
> | $N$ | Number of images in minibatch |
> | $C$ | Number of input feature maps |
> | $H$ | Height of input image |
> | $W$ | Width of input image |
> | $K$ | Number of output feature maps |
> | $R$ | Height of filter |
> | $S$ | Width of filter |
> | $u$ | Vertical Stride |
> | $v$ | Horizontal Stride |
> | ${pad\_h}$ | Height of zero padding |
> | ${pad\_w}$ | Width of zero padding | 

So, for the convolution; 
- Input Tensor $(D)$ : $N \times C \times H \times W$ - minibatch of images 
- Filter Tensor $(F)$ : $K \times C \times R \times S$ - convolution filters
- Output Tensor $(O)$ : $N \times K \times P \times Q$, where $P$ and $Q$ depend on the input size, filter size, stride and padding.

> [!note]  
> **Supported Algorithms**: GEMM, Winograd, and FFT – each with its own strengths!

### Implementation: 

- ***Step 1:***
    Initialize CUDNN with a handle:

    ```cpp
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    ```
- ***Step 2:***
    Describe the Tensors

    Define the input, filter and output:
    ```cpp
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, K, C, R, S);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, P, Q);
    ```
- ***Step 3:***
    Configure the Convolution

    Set padding, stride and mode:
    ```cpp
    cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, u, v, 1, 1, CUDNN_CROSS_CORRELATION);
    ```
- ***Step 4:***
    Pick an Algorithm

    Let cuDNN choose; or go manual
    ```cpp
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc, output_desc, 
                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    ```
- ***Step 5:***
    Allocate the GPU Memory

    Reserve space to the tensors
    ```cpp
    float *input_data, *filter_data, *output_data;
    cudaMalloc(&input_data, N * C * H * W * sizeof(float));
    cudaMalloc(&filter_data, K * C * R * S * sizeof(float));
    cudaMalloc(&output_data, N * K * P * Q * sizeof(float));
    ```
- ***Step 6:***
    Run the Convolution

    Execute the forward pass:
    ```cpp
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle, &alpha, input_desc, input_data, filter_desc, filter_data, 
                        conv_desc, algo, workspace, workspace_size, &beta, output_desc, output_data);  
    ```

- ***Step 7:***
    Wrapping up:

    Freeing Memory and Cleanup:
    ```cpp
    cudaFree(input_data);
    cudaFree(filter_data);
    cudaFree(output_data);
    cudnnDestroy(handle);
    ```
> [!tip]
> Start small – try $N=1,C=1,H=5,W=5$ with a $3 \times 3$ filter. Watch the output shrink to $3 \times 3$ (no padding)!

> [!note]
> Beyond GEMM: Winograd and FFT
> cuDNN doesn’t stop at GEMM: it’s got more tricks up its sleeve:
> - **Winograd**: A slick math shortcut that cuts down multiplications for small filters (like $3\times3$). Less work, same result—perfect for CNNs!
>
> - **FFT-Based**: For bigger filters, cuDNN switches to the Fast Fourier Transform, doing convolution in the frequency domain. It’s like solving a puzzle in a simpler way, then flipping it back.
> 
> ***Even better?*** cuDNN can pick the best algorithm for us based on our input and hardware, *or we can take the wheel and choose manually*.

> [!important]
> **TL;DR**
>
> When do we use what?  
> - **Winograd**: Awesome for small filters (e.g., $3 \times 3$).  
> - **FFT**: Shines with larger filters, though it eats more memory.  
> - **GEMM**: The reliable go-to, especially if memory’s tight.

> [Click Here](./cuDNN_convolution.cu) to redirect to the code.
