import triton 
import triton.language as tl 

import torch 

'''
Small theory section before diving into code:

How Batch Norm Works:
Batch Normalization (BatchNorm) is a trick to make neural nets train faster and stabler. It normalizes the inputs to a layer so they’ve got a mean of 0 and variance of 1 across the batch, then scales and shifts them with learnable params (gamma and beta). Here’s the breakdown:
- For a batch of data [batch_size, features], compute the mean (μ) and variance (σ²) per feature across the batch.
- Normalize each value: (x - μ) / √(σ² + ε), where ε (small constant like 1e-5) avoids division by zero.
- Scale and shift: γ * normalized + β, where γ and β are trained to tweak the output.
Why it’s dope: Keeps activations from exploding or vanishing, speeds up gradient flow, and cuts down on funky initialization hacks. In ML/DL/NLP, it’s everywhere—CNNs, transformers, you name it. Triton’s perfect for this ‘cause we can parallelize the stats and normalization across batch or features.

(I'll put  this in my Readme as well refer from there if this is not readable!)
'''

DEVICE = "cuda"

##@ Kernel to compute mean and variance per feature across batch
@triton.jit
def batchnorm_stats_kernel(input_ptr, mean_ptr, var_ptr, batch_size: tl.constexpr, features: tl.constexpr, stride_batch: tl.constexpr):
    feature_idx = tl.program_id(0)  ##! One block per feature column
    if feature_idx < features:
        # Offsets for this feature across all batch items
        offsets = tl.arange(0, batch_size) * stride_batch + feature_idx  #@@@ Grab all batch values for this feature
        x = tl.load(input_ptr + offsets, mask=tl.arange(0, batch_size) < batch_size)  ##@ Load ‘em up
        mean = tl.sum(x) / batch_size  #@@@ Mean = sum(x) / N
        x_minus_mean = x - mean
        var = tl.sum(x_minus_mean * x_minus_mean) / batch_size  ##! Var = E[(x - μ)²]
        tl.store(mean_ptr + feature_idx, mean)
        tl.store(var_ptr + feature_idx, var)

##@ Kernel to normalize using precomputed stats
@triton.jit
def batchnorm_norm_kernel(input_ptr, output_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, batch_size: tl.constexpr, features: tl.constexpr, stride_batch: tl.constexpr, eps: tl.constexpr):
    batch_idx = tl.program_id(0)  ##! One block per batch row
    if batch_idx < batch_size:
        offsets = batch_idx * stride_batch + tl.arange(0, features)  #@@@ Row-wise offsets
        mask = tl.arange(0, features) < features
        x = tl.load(input_ptr + offsets, mask=mask)  ##@ Load this batch item’s features
        mean = tl.load(mean_ptr + tl.arange(0, features), mask=mask)
        var = tl.load(var_ptr + tl.arange(0, features), mask=mask)
        gamma = tl.load(gamma_ptr + tl.arange(0, features), mask=mask)  #@@@ Scale factor
        beta = tl.load(beta_ptr + tl.arange(0, features), mask=mask)  #@@@ Shift factor
        norm = (x - mean) / tl.sqrt(var + eps) * gamma + beta  ##! The BatchNorm formula
        tl.store(output_ptr + offsets, norm, mask=mask)

##@ BatchNorm function to tie it all together
def batch_norm(input_tensor, gamma, beta, eps=1e-5):
    assert input_tensor.dim() == 2, "Input must be [batch_size, features]"
    batch_size, features = input_tensor.shape
    stride_batch = features  #@@@ Stride = num features (row length)

    mean = torch.zeros(features, device=DEVICE)
    var = torch.zeros(features, device=DEVICE)
    grid_stats = (features,)  ##! Grid = one block per feature
    batchnorm_stats_kernel[grid_stats](input_ptr=input_tensor, mean_ptr=mean, var_ptr=var, batch_size=batch_size, features=features, stride_batch=stride_batch)

    output = torch.empty_like(input_tensor)
    grid_norm = (batch_size,)  #@@@ Grid = one block per batch item
    batchnorm_norm_kernel[grid_norm](input_ptr=input_tensor, output_ptr=output, mean_ptr=mean, var_ptr=var, gamma_ptr=gamma, beta_ptr=beta, batch_size=batch_size, features=features, stride_batch=stride_batch, eps=eps)

    return output

if __name__ == "__main__":
    batch_size, features = 64, 128  #@@@ Like a small layer’s output
    input = torch.randn(batch_size, features, device=DEVICE)
    gamma = torch.ones(features, device=DEVICE)  ##! Scale = 1 for now
    beta = torch.zeros(features, device=DEVICE)  ##@ Shift = 0 for now
    
    triton_output = batch_norm(input, gamma, beta)
    
    bn = torch.nn.BatchNorm1d(features, affine=True, eps=1e-5).to(DEVICE)
    bn.weight.data = gamma  #@@@ Match γ
    bn.bias.data = beta  #@@@ Match β
    torch_output = bn(input)
    
    print("Input (first 2 rows):\n", input[:2])
    print("Triton Output (first 2):\n", triton_output[:2])
    print("PyTorch Output (first 2):\n", torch_output[:2])
    assert torch.allclose(triton_output, torch_output, atol=1e-5), "BatchNorm failed!"  #@@@ Verify
    print("Verification passed!")