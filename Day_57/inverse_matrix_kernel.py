import triton
import triton.language as tl
import torch

DEVICE = "cuda"

#@@@ Kernel to compute determinants for a batch of 2x2 matrices
@triton.jit
def det_kernel(input_ptr, det_ptr, batch_size, stride_batch, rows, cols):
    batch_idx = tl.program_id(0)  ##! Each thread block handles one matrix in the batch
    if batch_idx < batch_size:  #@@@ Make sure we don’t go out of bounds
        offset = batch_idx * stride_batch  ##@ Offset = how many floats to skip per matrix (4 for 2x2)
        # Load the 4 elements of a 2x2 matrix
        a = tl.load(input_ptr + offset + 0)  # [0,0]
        b = tl.load(input_ptr + offset + 1)  # [0,1]
        c = tl.load(input_ptr + offset + cols)  # [1,0]
        d = tl.load(input_ptr + offset + cols + 1)  # [1,1]
        det = a * d - b * c  ##! Classic 2x2 det formula: ad - bc
        tl.store(det_ptr + batch_idx, det)  #@@@ Save det for this matrix

##@ Kernel to compute inverses using the determinants
@triton.jit
def inverse_kernel(input_ptr, output_ptr, det_ptr, batch_size, stride_batch, rows, cols):
    batch_idx = tl.program_id(0)  #@@@ One thread per matrix, parallel across batch
    if batch_idx < batch_size:
        offset_in = batch_idx * stride_batch  ##! Input offset for this matrix
        offset_out = batch_idx * stride_batch  ##@ Output offset (same layout)
        det = tl.load(det_ptr + batch_idx)  #@@@ Grab the precomputed det
        # Handle singular matrices (det near 0) by setting inv_det to 0
        inv_det = tl.where(tl.abs(det) > 1e-6, 1.0 / det, 0.0)  ##! Avoid div-by-zero
        # Load input matrix elements
        a = tl.load(input_ptr + offset_in + 0)
        b = tl.load(input_ptr + offset_in + 1)
        c = tl.load(input_ptr + offset_in + cols)
        d = tl.load(input_ptr + offset_in + cols + 1)
        # Write inverse: [d, -b; -c, a] scaled by 1/det
        tl.store(output_ptr + offset_out + 0, d * inv_det)  #@@@ [0,0] gets d/det
        tl.store(output_ptr + offset_out + 1, -b * inv_det)  # [0,1]
        tl.store(output_ptr + offset_out + cols, -c * inv_det)  # [1,0]
        tl.store(output_ptr + offset_out + cols + 1, a * inv_det)  # [1,1]

#@@@ Main function to invert a batch of 2x2 matrices
def batch_matrix_inverse_2x2(input_tensor):
    assert input_tensor.dim() == 3 and input_tensor.shape[1:] == (2, 2), "Input must be batch of 2x2 matrices"
    batch_size, rows, cols = input_tensor.shape  ##! Unpack dims: [batch, 2, 2]
    stride_batch = rows * cols  ##@ Stride = 4 (2*2) floats per matrix

    # Compute determinants in parallel
    det_output = torch.zeros(batch_size, device=DEVICE)  #@@@ One det per matrix
    grid = (batch_size,)  ##! Grid size = number of matrices
    det_kernel[grid](input_tensor, det_output, batch_size, stride_batch, rows, cols)

    # Compute inverses
    output = torch.zeros_like(input_tensor)  ##@ Zero-init so singular matrices are clean
    inverse_kernel[grid](input_tensor, output, det_output, batch_size, stride_batch, rows, cols)

    # Warn about singular matrices
    singular_count = torch.sum(torch.abs(det_output) < 1e-6).item()  #@@@ Count dets too small
    if singular_count > 0:
        print(f"Warning: {singular_count} matrices are singular (det < 1e-6), inverses set to zero.")

    return output

if __name__ == "__main__":
    batch_size = 1000  #@@@ Testing with 1000 matrices
    input = torch.randn(batch_size, 2, 2, device=DEVICE)  ##! Random batch of 2x2s
    
    # Triton inverse
    triton_output = batch_matrix_inverse_2x2(input)
    torch_output = torch.inverse(input)  ##@ PyTorch’s version for checking
    
    # Print a sample
    idx = 0
    print(f"\nSample Matrix [{idx}]:\n", input[idx])
    print(f"Triton Inverse [{idx}]:\n", triton_output[idx])
    print(f"PyTorch Inverse [{idx}]:\n", torch_output[idx])
    
    # Verify
    assert torch.allclose(triton_output, torch_output, atol=1e-5, equal_nan=True), "Batch inverse failed!"  #@@@ Check all match
    print("Verification passed!")

    # Quick perf test
    import time
    torch.cuda.synchronize()  ##! Sync GPU before timing
    start = time.time()
    for _ in range(10):
        triton_output = batch_matrix_inverse_2x2(input)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        torch_output = torch.inverse(input)
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    print(f"\nTriton Time: {triton_time:.4f} seconds")  #@@@ How fast we go
    print(f"PyTorch Time: {torch_time:.4f} seconds")