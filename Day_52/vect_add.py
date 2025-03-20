import triton
import triton.language as tl

## First let's learn vector addition 

@triton.jit #@ triton.jit decorator is used to define Triton kernels
def add_kernel(X, Y, Z, N):
    pid = tl.program_id(0)
    idx = pid * 512 + tl.arange(0, 512) #@ 512 is block size
    mask = idx < N
    x = tl.load(X + idx, mask=mask)
    y = tl.load(Y + idx, mask=mask)
    tl.store(Z + idx, x + y, mask=mask)
    
## Example usage
import torch 
N = 1024
X = torch.randn(N, device='cuda')
Y = torch.randn(N, device='cuda')
Z = torch.empty(N, device='cuda')
add_kernel[(N // 512,)](X, Y, Z, N)

torch.cuda.synchronize()  # Ensuring all GPU computations are complete

# Verify correctness
print("Resulting Tensor Z:", Z)
Z_torch = X + Y
print("PyTorch Result:", Z_torch)
print("Maximum Difference:", torch.max(torch.abs(Z - Z_torch)))
     