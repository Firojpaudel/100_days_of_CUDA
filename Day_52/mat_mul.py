import torch 

import triton 
import triton.language as tl

DEVICE = "cuda"

@triton.jit
def matmul_kernel(
    ##@ Pointers to matrices
    A, B, C,
    ##@ Matrix dimensions
    M, N, K, 
    ##@ Strides
    stride_am, stride_ak, 
    stride_bk, stride_bn, 
    stride_cm, stride_cn, 
    BLOCK_SIZE: tl.constexpr
):
    # Program Ids
    pid_m = tl.program_id(0) #@ Row Block ID
    pid_n = tl.program_id(1) #@ Column BlocK ID
       
    # Offsets for A and B matrices
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_k = tl.arange(0, BLOCK_SIZE)
    
    # Pointers to blocks of A and B
    A_block_ptr = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_block_ptr = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator for C
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype = tl.float32)
    
    # Loop over K dimension in blocks: 
    for k in range(0, K, BLOCK_SIZE):
        ##@ Load block of A with masking
        A_block = tl.load(A_block_ptr, mask= (offs_m[:, None] < M) & ((k+ offs_k[None, :]) < K), other= 0.0)
        '''
        So, first I used offs_k[:, None] < K since offs_k = tl.arange(0, BLOCK_SIZE), now this would check if the column offsets within the block are less than k. But, here's a catch: it doesn't account for where the block starts in the matrix, which is controlled by k.
        '''
        ##@ Load block of B with masking
        B_block = tl.load(B_block_ptr, mask= ((k+ offs_k[:, None]) < K) & (offs_n[None, :] < N), other= 0.0)
        
        ##@ Matrix multiplication using dot product
        acc += tl.dot(A_block, B_block)
        
        ##@ Move to the next K block
        A_block_ptr += BLOCK_SIZE * stride_ak
        B_block_ptr += BLOCK_SIZE * stride_bk
        
    # Writing all this result to C
    C_block_ptr = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_block_ptr, acc, mask= (offs_m[:, None] < M) & (offs_n[None, :] < N))

def matmul(a,b):
    ## Getting the matrix dimensions 
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "inner dimensions must match" ##@ Verifying the inner dimensions match 
    
    c = torch.empty((M,N), device=DEVICE, dtype=torch.float32)
    
    BLOCK_SIZE = 32
    
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE)) ##@ cdiv= ceiling division
    
    matmul_kernel[grid](
        a, b, c, 
        M, N, K, 
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE= BLOCK_SIZE
    )
    
    return c

# The main function to run 
if __name__ == "__main__":
    M, K, N = 128, 256, 128
    a= torch.randn((M,K), device=DEVICE)
    b= torch.randn((K,N), device=DEVICE)
    
    c_triton = matmul(a,b)
    print(f"Triton Multiplication Output: {c_triton}")
    
    c_torch = torch.matmul(a,b)
    print(f"PyTorch Multiplication Output: {c_torch}")
    
    diff = torch.max(torch.abs(c_triton - c_torch))
    print(f"Maximum Difference: {diff}")