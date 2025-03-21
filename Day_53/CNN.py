import torch
import time 
import triton 
import triton.language as tl 

import numpy as np 

##@ So first lets define convolution kernel 
@triton.jit
def conv_kernel(
    input_ptr, filter_ptr, output_ptr,
    H, W, K, H_out, W_out, bias, 
    stride_h: tl.constexpr, stride_w: tl.constexpr,
):
    ##@ Output position for this kernel 
    pid_h = tl.program_id(0)  # Row index
    pid_w = tl.program_id(1) #Column Index 
    
 
    # Check if thread is within output bounds
    if pid_h >= H_out or pid_w >= W_out:
        return
        
    ##@ Accumulation of convolution result
    sum = 0.0
    for p in range(K):  # Filter height
        for q in range(K):  # Filter width (K x K kernel)
            h_indx = pid_h * stride_h + p  # Input height index
            w_indx = pid_w * stride_w + q  # Input width index
            in_bounds = (h_indx < H) & (w_indx < W)
            if in_bounds:
                input_indx = h_indx * W + w_indx ## Flattening (row major order)
                filter_indx = p * K + q ## Flattening (row major order)
                sum += tl.load(input_ptr + input_indx) * tl.load(filter_ptr + filter_indx) ## Dot product accumulation 
    
    ##@ Now writing the output with bias:
    output_indx = pid_h * W_out + pid_w  ## Faltten -> 1D 
    tl.store(output_ptr + output_indx, sum + bias) 
    

## Wrapper to launch the kernel 
def conv_layer_triton (input, filter, bias):
    H, W = input.shape
    K = filter.shape[0] ## Assuming a square kernel filter
    H_out = H- K + 1 # Consult the docs I've written in my CUDA implementation for this  
    W_out = W - K + 1 # Same here
    
    ## Converting to torch tensors and moving to GPU
    input = torch.from_numpy(input).cuda()
    filter = torch.from_numpy(filter).cuda()
    output = torch.zeros(H_out, W_out, dtype=torch.float32).cuda()
    
    ##@ Define grid 
    grid = (H_out, W_out)
    
    ##@ Calling  the kernel 
    conv_kernel[grid](
        input, filter, output, 
        H, W, K, H_out, W_out, bias,
        stride_h=1, stride_w=1,
    )
    return output.cpu().numpy()
    
##@ The sigmoid function 
def sigmoid(x):
    return 1/ (1+ np.exp(-x))

##@ Subsampling Layer:
def subsampling_layer(input, K):
    H, W = input.shape
    H_out = H//K 
    W_out = W//K
    
    output = np.zeros((H_out, W_out), dtype=np.float32)
    
    ##@ Pooling and average
    for h in range(H_out):
        for w in range(W_out):
            region = input[h*K:(h+1)*K, w*K:(w+1)*K]
            output[h, w] = np.mean(region)
            
    return sigmoid(output)

## The main function 
if __name__ == "__main__":
    input = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1]
    ], dtype=np.float32)
    
    filter = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ], dtype=np.float32)
    
    bias = 0.01
    
    print("Input:\n", input)
    print("Filter:\n", filter)
    print("Bias:", bias)
    
    # Run convolution
    start_time = time.perf_counter()
    conv_output = conv_layer_triton(input, filter, bias)
    conv_time = (time.perf_counter() - start_time) * 1000 # In ms
    print("After convolution:\n", conv_output)
    print(f"Convolution runtime: {conv_time:.6f} ms")
    
    # Run subsampling
    start_time = time.perf_counter()
    pool_output = subsampling_layer(conv_output, 2)
    pool_time = (time.perf_counter() - start_time) * 1000 # ms 
    print("After subsampling:\n", pool_output)
    print(f"Subsampling runtime: {pool_time:.6f} ms")        