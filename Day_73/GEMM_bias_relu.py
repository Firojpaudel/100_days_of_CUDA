import torch
import triton
import triton.language as tl

@triton.jit
def gemm_bias_relu_kernel(A_ptr, W_ptr, b_ptr, C_ptr,
                          B, N, M,
                          stride_a_batch, stride_a_feat,
                          stride_w_feat, stride_w_out,
                          stride_c_batch, stride_c_out,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ##@ Fetch program ID to identify the current block's position
    pid_m = tl.program_id(0)  ##@ Block ID in M dimension
    pid_n = tl.program_id(1)  ##@ Block ID in N dimension

    ##@ Compute the starting indices for the current block
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)

    ##@ Initialize the accumulator with high precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    ##@ Craft row and column vectors for broadcasting
    m_rows = offs_m[:, None]
    n_cols = offs_n[None, :]

    ##@ Iterate over k-dimension blocks
    for k in range(0, N, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_cols = offs_k[None, :]

        ##@ Load A block - shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + m_rows * stride_a_batch + k_cols * stride_a_feat
        a_mask = (offs_m[:, None] < B) & (offs_k[None, :] < N)
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)

        ##@ Load W block - shape (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + offs_k[:, None] * stride_w_feat + offs_n[None, :] * stride_w_out
        w_mask = (offs_k[:, None] < N) & (offs_n[None, :] < M)
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)
        ##@ Perform matrix multiplication
        acc += tl.dot(a_block, w_block)
    
    ##@ Load and add bias
    bias_ptrs = b_ptr + offs_n
    bias_mask = offs_n < M
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    
    ##@ Add bias to each row (broadcasting)
    acc = acc + bias[None, :]
    
    ##@ Apply ReLU activation
    acc = tl.maximum(acc, 0.0) ##@ make sure to activate it
    
    ##@ Store output
    c_mask = (offs_m[:, None] < B) & (offs_n[None, :] < M)
    c_ptrs = C_ptr + offs_m[:, None] * stride_c_batch + offs_n[None, :] * stride_c_out
    tl.store(c_ptrs, acc, mask=c_mask)

def gemm_bias_relu_triton(A, W, b):
    ##@ Extract dimensions
    B, N = A.shape
    M, N_check = W.shape

    ##@ Ensure inner dimensions align
    assert N == N_check, f"Inner dimensions must match: {N} vs {N_check}"

    ##@ Create output tensor
    C = torch.empty((B, M), device=A.device, dtype=torch.float32)

    ##@ Define block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32

    ##@ Compute grid dimensions
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(M, BLOCK_N))

    ##@ Launch kernel
    gemm_bias_relu_kernel[grid](
        A, W, b, C,
        B, N, M,
        A.stride(0), A.stride(1),
        W.stride(1), W.stride(0),  ##@ Transposed access for W
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )

    return C

##@ === Main test ===
if __name__ == "__main__":
    ##@ Define problem dimensions
    B, N, M = 64, 128, 256

    ##@ Generate random input data
    torch.manual_seed(0)  ##@ For reproducibility
    A = torch.randn((B, N), device='cuda', dtype=torch.float32)
    W = torch.randn((M, N), device='cuda', dtype=torch.float32)
    b = torch.randn((M,), device='cuda', dtype=torch.float32)

    ##@ Compute with Triton kernel
    C_triton = gemm_bias_relu_triton(A, W, b)

    ##@ Compute reference result with PyTorch
    C_ref = torch.relu(A @ W.T + b)

    ##@ Verify results
    print("✅ Output shape:", C_triton.shape)
    print("✅ Sample values (Triton):", C_triton[0, :5])
    print("✅ Sample values (PyTorch):", C_ref[0, :5])

    ##@ Check for numerical correctness
    max_diff = (C_triton - C_ref).abs().max().item()
    mean_diff = (C_triton - C_ref).abs().mean().item()
    all_close = torch.allclose(C_triton, C_ref, atol=1e-3)
    
    print("✅ All close:", all_close)
    print("✅ Max absolute error:", max_diff)
    print("✅ Mean absolute error:", mean_diff)
    ##@ More detailed comparison if needed
    if not all_close:
        diff = (C_triton - C_ref).abs()
        worst_idx = diff.argmax().item()
        worst_i, worst_j = worst_idx // M, worst_idx % M
        print(f"Largest difference at [{worst_i}, {worst_j}]:")
        print(f"  Triton: {C_triton[worst_i, worst_j].item()}")
        print(f"  PyTorch: {C_ref[worst_i, worst_j].item()}")