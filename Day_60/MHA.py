import triton
import triton.language as tl
import math
import torch
import torch.nn.functional as F
import time #@ Benchmarking....

DEVICE = "cuda"

##@ Query, Key and Values kernel
@triton.jit
def qkv_kernel(
    input_ptr, wq_ptr, wk_ptr, wv_ptr, bq_ptr, bk_ptr, bv_ptr, q_ptr, k_ptr, v_ptr,
    batch_size: tl.constexpr, seq_len: tl.constexpr, embed_dim: tl.constexpr,
    head_dim: tl.constexpr, num_heads: tl.constexpr, stride_batch: tl.constexpr,
    stride_seq: tl.constexpr, stride_head: tl.constexpr
):
    batch_idx = tl.program_id(0) #! One block per batch 
    seq_idx = tl.program_id(1) #! One thread per sequence position 
    head_idx = tl.program_id(2) # Added for head dimension
    
    if batch_idx >= batch_size: 
        return
    if seq_idx >= seq_len:  
        return
    if head_idx >= num_heads:
        return
        
    input_offset = batch_idx * stride_batch + seq_idx * stride_seq
    qkv_offset = batch_idx * stride_batch + seq_idx * stride_seq + head_idx * stride_head
    
    #@ Projection Q, K, V
    for d in range(head_dim):
        acc_q = tl.load(bq_ptr + head_idx * head_dim + d) #! Include bias for query
        acc_k = tl.load(bk_ptr + head_idx * head_dim + d) #! Include bias for key
        acc_v = tl.load(bv_ptr + head_idx * head_dim + d) #! Include bias for value
        #@ Matmul in Triton a'ight! 
        for e in range(embed_dim):
            x_val = tl.load(input_ptr + input_offset + e) 
            wq_val = tl.load(wq_ptr + (head_idx * head_dim + d) * embed_dim + e)  
            wk_val = tl.load(wk_ptr + (head_idx * head_dim + d) * embed_dim + e)  
            wv_val = tl.load(wv_ptr + (head_idx * head_dim + d) * embed_dim + e)  
            acc_q += x_val * wq_val
            acc_k += x_val * wk_val
            acc_v += x_val * wv_val
        tl.store(q_ptr + qkv_offset + d, acc_q) #! Projection of Q
        tl.store(k_ptr + qkv_offset + d, acc_k) #! Projection of K
        tl.store(v_ptr + qkv_offset + d, acc_v) #! Projection of V

##@ Kernel for attention scores and output:
@triton.jit 
def attention_kernel(
    q_ptr, k_ptr, v_ptr, scores_ptr, output_ptr, batch_size: tl.constexpr,
    seq_len: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr,
    stride_batch: tl.constexpr, stride_seq: tl.constexpr, stride_head: tl.constexpr,
    scale: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    if batch_idx >= batch_size:
        return 
    if seq_idx >= seq_len: 
        return
    if head_idx >= num_heads:
        return
    
    q_offset = batch_idx * stride_batch + seq_idx * stride_seq + head_idx * stride_head
    scores_offset = batch_idx * seq_len * num_heads * seq_len + seq_idx * num_heads * seq_len + head_idx * seq_len
    
    #@ Compute Q * K^T
    for k_seq in range(seq_len):
        k_offset = batch_idx * stride_batch + k_seq * stride_seq + head_idx * stride_head
        mul = 0.0
        for d in range(head_dim):
            q_val = tl.load(q_ptr + q_offset + d)
            k_val = tl.load(k_ptr + k_offset + d)
            mul += q_val * k_val
        tl.store(scores_ptr + scores_offset + k_seq, mul * scale)
    
    #@ Softmax
    scores_base = scores_ptr + scores_offset
    # Compute softmax explicitly
    max_score = float('-inf')
    for k_seq in range(seq_len):
        score = tl.load(scores_base + k_seq)
        max_score = tl.maximum(max_score, score)
    exp_sum = 0.0
    for k_seq in range(seq_len):
        score = tl.load(scores_base + k_seq)
        exp_score = tl.exp(score - max_score)
        tl.store(scores_base + k_seq, exp_score)
        exp_sum += exp_score
    for k_seq in range(seq_len):
        exp_score = tl.load(scores_base + k_seq)
        tl.store(scores_base + k_seq, exp_score / exp_sum)
    
    out_offset = batch_idx * stride_batch + seq_idx * stride_seq + head_idx * stride_head
    for d in range(head_dim):
        acc = 0.0
        for k_seq in range(seq_len):
            v_offset = batch_idx * stride_batch + k_seq * stride_seq + head_idx * stride_head
            v_val = tl.load(v_ptr + v_offset + d)
            attn_weight_k = tl.load(scores_base + k_seq)
            acc += attn_weight_k * v_val
        tl.store(output_ptr + out_offset + d, acc)

##@ Now defining the wrapper function
def multi_head_attention(input_tensor, attention_layer):
    assert input_tensor.dim() == 3, "Input must be [batch_size, seq_len, embed_dim]"
    batch_size, seq_len, embed_dim = input_tensor.shape
    num_heads = attention_layer.num_heads
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    head_dim = embed_dim // num_heads
    stride_batch = seq_len * num_heads * head_dim
    stride_seq = num_heads * head_dim
    stride_head = head_dim
    
    # Scale computation 
    scale = 1.0 / math.sqrt(head_dim)
    
    # Use the provided attention_layer for weights and biases
    wq = attention_layer.in_proj_weight[:embed_dim].T  # Direct use, no cloning
    wk = attention_layer.in_proj_weight[embed_dim:2*embed_dim].T
    wv = attention_layer.in_proj_weight[2*embed_dim:].T
    bq = attention_layer.in_proj_bias[:embed_dim] if attention_layer.in_proj_bias is not None else torch.zeros(embed_dim, device=DEVICE)
    bk = attention_layer.in_proj_bias[embed_dim:2*embed_dim] if attention_layer.in_proj_bias is not None else torch.zeros(embed_dim, device=DEVICE)
    bv = attention_layer.in_proj_bias[2*embed_dim:] if attention_layer.in_proj_bias is not None else torch.zeros(embed_dim, device=DEVICE)
    wo = attention_layer.out_proj.weight
    bo = attention_layer.out_proj.bias if attention_layer.out_proj.bias is not None else torch.zeros(embed_dim, device=DEVICE)

    # Output tensors
    q = torch.empty(batch_size, seq_len, num_heads, head_dim, device=DEVICE)
    k = torch.empty(batch_size, seq_len, num_heads, head_dim, device=DEVICE)
    v = torch.empty(batch_size, seq_len, num_heads, head_dim, device=DEVICE)
    scores = torch.zeros(batch_size, seq_len, num_heads, seq_len, device=DEVICE, dtype=torch.float32)
    output = torch.empty(batch_size, seq_len, num_heads, head_dim, device=DEVICE)

    # QKV projection with bias
    grid_qkv = (batch_size, seq_len, num_heads)
    qkv_kernel[grid_qkv](
        input_tensor, wq, wk, wv, bq, bk, bv, q, k, v,
        batch_size, seq_len, embed_dim, head_dim, num_heads,
        stride_batch, stride_seq, stride_head
    )

    # Attention computation
    grid_attn = (batch_size, seq_len, num_heads)
    attention_kernel[grid_attn](
        q, k, v, scores, output, batch_size, seq_len, head_dim, num_heads,
        stride_batch, stride_seq, stride_head, scale
    )

    # Apply output projection
    output = output.view(batch_size * seq_len, embed_dim)
    output = torch.nn.functional.linear(output, wo, bo)
    output = output.view(batch_size, seq_len, embed_dim)

    return output

if __name__ == "__main__":
    torch.manual_seed(42)  #!! For reproducibility
    batch_size, seq_len, embed_dim = 2, 8, 16
    num_heads = 4
    input = torch.randn(batch_size, seq_len, embed_dim, device=DEVICE)
    
    # Creating a single attention layer to share weights
    attention = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(DEVICE)
    
    # Warm-up runs
    for _ in range(10):
        _ = multi_head_attention(input, attention)
        _ = attention(input, input, input, need_weights=False)
    torch.cuda.synchronize()

    # Benchmark Triton
    triton_times = []
    for _ in range(1000):
        start = time.time()
        triton_output = multi_head_attention(input, attention)
        torch.cuda.synchronize()  # Wait for GPU to finish
        triton_times.append(time.time() - start)
    triton_avg = sum(triton_times) / len(triton_times) * 1000  # Convert to ms
    
    # Benchmark PyTorch
    pytorch_times = []
    for _ in range(1000):
        start = time.time()
        torch_output, _ = attention(input, input, input, need_weights=False)
        torch.cuda.synchronize()  # Wait for GPU to finish
        pytorch_times.append(time.time() - start)
    pytorch_avg = sum(pytorch_times) / len(pytorch_times) * 1000  # Convert to ms
    
    # Verify correctness
    max_diff = torch.max(torch.abs(triton_output - torch_output))
    
    ##@ Displayin'
    print("Input: \n", input[0])
    print("Triton Output: \n", triton_output[0])
    print("PyTorch Output: \n", torch_output[0])
    print("Max difference: ", max_diff)
    print(f"Triton average time: {triton_avg:.4f} ms")
    print(f"PyTorch average time: {pytorch_avg:.4f} ms")
    print(f"Speedup (PyTorch/Triton): {pytorch_avg / triton_avg:.2f}x")