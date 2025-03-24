import triton 
import triton.language as tl 

import torch 

DEVICE = "cuda"

##@ Triton kernel for ReLU 
@triton.jit 
def relu_function_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis= 0) #! Getting the program ID (thread block index)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets< n_elements ##@ Preventin outof bounds 
    x= tl.load(input_ptr + offsets, mask = mask)
    y= tl.where(x>0, x, 0) #@ ReLU implemented here --> if x> 0; keep x else 0
    tl.store(output_ptr + offsets, y, mask = mask) ##@ Storingggg
    
#@ kernel run 
def relu(input_tensor):
    output = torch.empty_like(input_tensor, device=DEVICE)
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_function_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE = BLOCK_SIZE)
    return output

##@ Now testing 
if __name__ == "__main__":
    input= torch.randn(5120, device=DEVICE)
    output= relu(input)
    expected = torch.relu(input) #@ Torch inbuilt function check for comparison
    print("Input was: ", input[:10])
    print("Output is: ", output[:10])
    ##@ Checking the correctness 
    print("PyTorch Expected Values:", expected[:10])
    