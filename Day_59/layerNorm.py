import triton 
import triton.language as tl 

import torch 

DEVICE = "cuda"

##@ Kernel to compute mean and variance per sample accross features 
@triton.jit
def layernorm_stats_kernel (input_ptr, mean_ptr, var_ptr, batch_size: tl.constexpr, features: tl.constexpr, stride_batch: tl.constexpr):
    batch_idx = tl.program_id(0) #! One block per element
    if batch_idx < batch_size:
        offsets = batch_idx*stride_batch + tl.arange(0, features) #! All features
        mask = tl.arange(0, features) < features
        x= tl.load(input_ptr + offsets, mask= mask) #! Loading Samples with the mask 
        mean = tl.sum(x) / features #! Mean over the features
        x_minus_mean =  x - mean #! variance = mean(x-mean)^2 ~ in terms of expectation
        var = tl.sum(x_minus_mean*x_minus_mean) / features #! Variance
        #@ Storing 
        tl.store(mean_ptr + batch_idx, mean)
        tl.store(var_ptr + batch_idx, var)
    
##@ Kernel to Noramlize using above computed stats
@triton.jit
def layernorm_norm_kernel(input_ptr, output_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, batch_size: tl.constexpr, features: tl.constexpr, stride_batch: tl.constexpr, eps: tl.constexpr):
    batch_idx = tl.program_id(0)  ##! One block per sample
    if batch_idx < batch_size:
        offsets = batch_idx * stride_batch + tl.arange(0, features)  #! Row-wise offsets
        mask = tl.arange(0, features) < features
        x = tl.load(input_ptr + offsets, mask=mask)  #! Load this sample’s features
        mean = tl.load(mean_ptr + batch_idx)  #! Scalar mean for this sample
        var = tl.load(var_ptr + batch_idx)  #! Scalar variance
        gamma = tl.load(gamma_ptr + tl.arange(0, features), mask=mask)  #! Feature-wise scale
        beta = tl.load(beta_ptr + tl.arange(0, features), mask=mask)  #! Feature-wise shift
        norm = (x - mean) / tl.sqrt(var + eps) * gamma + beta  #! LayerNorm formula
        ##@ The formula is y = (x- E(x))/sqrt(var(x) + epsilon) * gamma + beta
        tl.store(output_ptr + offsets, norm, mask=mask)

#@ LayerNorm Kernel function wrapper
def layer_norm(input_tensor, gamma, beta, eps= 1e-5):
    #! The input tensor must be of dimension of [batch_size, features]
    assert input_tensor.dim() == 2, "Invalid dimension"
    batch_size, features = input_tensor.shape
    stride_batch = features
    
    #!! Computing the stats
    mean = torch.zeros(batch_size, device=DEVICE)
    var = torch.zeros(batch_size, device=DEVICE)
    
    #! Deifining the grid
    grid = (batch_size, ) #! Only one block_per_sample
    layernorm_stats_kernel[grid](input_tensor, mean, var, batch_size, features, stride_batch)
    
    #@ Normalizinggg
    output = torch.empty_like(input_tensor)
    layernorm_norm_kernel[grid](input_tensor, output, mean, var, gamma, beta, batch_size, features, stride_batch, eps)
    return output


if __name__ == "__main__":
    batch_size, features = 64, 128  #! Like a transformer's hidden state
    input = torch.randn(batch_size, features, device=DEVICE)
    gamma = torch.ones(features, device=DEVICE)  ##! Scale = 1
    beta = torch.zeros(features, device=DEVICE)  ##! Shift = 0
    
    # Triton LayerNorm
    triton_output = layer_norm(input, gamma, beta)
    
    # PyTorch LayerNorm
    ln = torch.nn.LayerNorm(features, eps=1e-5).to(DEVICE)
    ln.weight.data = gamma  #@@@ Matching gamma
    ln.bias.data = beta  #@@@ Matching beta
    torch_output = ln(input)
    
    print("Input (first 2):\n", input[:2])
    print("Triton Output (first 2):\n", triton_output[:2])
    print("PyTorch Output (first 2):\n", torch_output[:2])
    assert torch.allclose(triton_output, torch_output, atol=1e-5), "LayerNorm failed!"  
    print("Verification passed!")