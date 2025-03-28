import triton
import triton.language as tl
import math
import torch

DEVICE = "cuda"

##@ Query, Key and Values kernel
@triton.jit
def qkv_kernel(
    input_ptr, wq_ptr, wk_ptr, wv_ptr, q_ptr, k_ptr, v_ptr, \
    batch_size: tl.constexpr, seq_len: tl.constexpr, embed_dim: tl.constexpr, \
    head_dim: tl.constexpr, stride_batch: tl.constexpr, stride_seq: tl.constexpr \
    ):
        batch_idx = tl.program_id(0) #! One block per batch 
        seq_idx = tl.program_id(1) #! One thread per sequence position 
        if batch_idx < batch_size:
            if seq_idx < seq_len:
                input_offset = batch_idx * stride_batch + seq_idx * stride_seq
                q_offset = input_offset
                k_offset = input_offset
                v_offset = input_offset
                
                #@ Projection Q, K, V
                for d in range(embed_dim):  # Corrected loop over embed_dim
                    acc_q = 0.0 #! Accumulator for query
                    acc_k = 0.0 #! Accumulator for key 
                    acc_v = 0.0 #! Accumulator for value 
                    #@ Matmul in Triton a'ight! 
                    for e in range(embed_dim):
                        x_val = tl.load(input_ptr + input_offset + e) 
                        wq_val = tl.load(wq_ptr + e * embed_dim + d)  # Corrected weight index
                        wk_val = tl.load(wk_ptr + e * embed_dim + d)  # Corrected weight index
                        wv_val = tl.load(wv_ptr + e * embed_dim + d)  # Corrected weight index
                        acc_q += x_val * wq_val
                        acc_k += x_val * wk_val
                        acc_v += x_val * wv_val
                    tl.store(q_ptr + q_offset + d, acc_q) #! Projection of Q
                    tl.store(k_ptr + k_offset + d, acc_k) #! Projection of K
                    tl.store(v_ptr + v_offset + d, acc_v) #! Projection of V

##@ Kernel for attention scores and output:
@triton.jit 
def attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr, batch_size: tl.constexpr, \
    seq_len: tl.constexpr, head_dim: tl.constexpr, num_heads: tl.constexpr, \
    stride_batch: tl.constexpr, stride_seq: tl.constexpr, stride_head: tl.constexpr, scale: tl.constexpr
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # Separate if statements to avoid chained boolean operators
    if batch_idx >= batch_size:
        return
    if seq_idx >= seq_len:
        return
    if head_idx >= num_heads:
        return
    
    q_offset = batch_idx * stride_batch + seq_idx * stride_seq + head_idx * stride_head
    scores = tl.zeros([seq_len], dtype=tl.float32) #! Temporary scores array!! 
    '''
    Our Formula:
            Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    '''
    #@ Time to compute Q * K^T
    for k_seq in range(seq_len):
        k_offset = batch_idx * stride_batch + k_seq * stride_seq + head_idx * stride_head  # Corrected offset
        mul = 0.0
        for d in range(head_dim):
            q_val = tl.load(q_ptr + q_offset + d)
            k_val = tl.load(k_ptr + k_offset + d)
            mul += q_val * k_val
        '''
        Here, k_seq loops over tokens (rows of (K)), while (d) loops over dimensions (columns of (K)) Hence mimicking K^T 
        '''
        scores[k_seq] = mul * scale
    
    #@ Sofmax 
    max_score = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - max_score)
    sum_exp = tl.sum(exp_scores, axis=0)
    attn_weights = exp_scores / sum_exp #! Softmax implemented here!! 
    out_offset = batch_idx * stride_batch + seq_idx * stride_seq + head_idx * stride_head  # Corrected offset
    for d in range(head_dim):
        acc = 0.0
        for k_seq in range(seq_len): #! Weighted sum with V
            v_offset = batch_idx * stride_batch + k_seq * stride_seq + head_idx * stride_head  # Corrected offset
            v_val = tl.load(v_ptr + v_offset + d)
            acc += attn_weights[k_seq] * v_val
        tl.store(output_ptr + out_offset + d, acc) #! Finally saving the attention output 
            
##@ Now defining the wrapper function
def multi_head_attention(input_tensor, num_heads=4):
    assert input_tensor.dim() == 3, "Input must be [batch_size, seq_len, embed_dim]"
    batch_size, seq_len, embed_dim = input_tensor.shape
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    head_dim = embed_dim // num_heads
    stride_batch = seq_len * embed_dim
    stride_seq = embed_dim
    stride_head = head_dim  # Corrected stride_head
    
    # Scale computation 
    scale = 1.0 / math.sqrt(head_dim)
    
    # Random weights for Q, K, V projections
    wq = torch.randn(embed_dim, embed_dim, device=DEVICE)
    wk = torch.randn(embed_dim, embed_dim, device=DEVICE)
    wv = torch.randn(embed_dim, embed_dim, device=DEVICE)

    # Output tensors
    q = torch.empty(batch_size, seq_len, embed_dim, device=DEVICE)  #@@@ Full Q tensor
    k = torch.empty(batch_size, seq_len, embed_dim, device=DEVICE)  #@@@ Full K tensor
    v = torch.empty(batch_size, seq_len, embed_dim, device=DEVICE)  #@@@ Full V tensor
    output = torch.empty(batch_size, seq_len, embed_dim, device=DEVICE)  #@@@ Attention output

    # QKV projection
    grid_qkv = (batch_size, seq_len)  #@@@ 2D grid: batch x seq
    qkv_kernel[grid_qkv](input_tensor, wq, wk, wv, q, k, v, batch_size, seq_len, embed_dim, head_dim, stride_batch, stride_seq)

    # Attention computation
    grid_attn = (batch_size, seq_len, num_heads)  # Corrected grid order
    attention_kernel[grid_attn](q, k, v, output, batch_size, seq_len, head_dim, num_heads, stride_batch, stride_seq, stride_head, scale)

    return output

if __name__ == "__main__":
    batch_size, seq_len, embed_dim = 2, 8, 16
    num_heads = 4
    input = torch.randn(batch_size, seq_len, embed_dim, device=DEVICE)
    
    #@ Triton Output 
    triton_output = multi_head_attention(input, num_heads)
    
    #@ PyTorch Attention 
    attention = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(DEVICE)
    torch_output, _ = attention(input, input, input, need_weights=False)
    
    ##@ Displayin'
    print("Input: \n", input[0])
    print("Triton Output: \n", triton_output[0])
    print("PyTorch Output: \n", torch_output[0])