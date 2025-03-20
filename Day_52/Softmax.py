import torch 

import triton
import triton.language as tl 
from triton.runtime import driver 

DEVICE = "cuda"

@triton.jit
def softmax_kernel (
    output_ptr, input_ptr, 
    input_row_stride, output_row_stride, 
    n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    #@ Creating the offsets for Columns 
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    input_ptrs = row_start_ptr + col_offsets
    
    # Loading the row into SRAM and masking for out-of-bounds accesses
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask= mask, other=-float('inf'))
    '''
    Explaning why I'm using negative infinity over here:
    So the softmax function is: 
                    S(x_i) = e^(x_i) / sum (e^x)
    When the input is infinity, its exp becomes 0. 
    
    So this ensures that out-of-bounds elements contribute ZERO to both the numerator and denominator. 
    '''
    
    ## Substracting the maximum for numerical stability
    '''
    Why we substract the maximum value?
    The softmax function involves computing the exponentials of input values. 
    
    And this well presents two major risks:
    1. Overflow Risk 
    2. Underflow Risk
    
    Overflow Risk: If any values in the input are large positive numbers computing e^(x_i) can easily exceed the maximum representable float value
    
    Underflow Risk: If some values are much smaller than others, their exponentials become extremely small numbers that might be rounded to zero due to limited precision. Hence, leading to inaccurate results or division by zero errors
    
    So the mathematical solution is to rewrite the mathematical equation of softmax using the mathematical identity:
    ie.,                    Softmax(x_i) = e^(x_i- C) / sum (e^x-C)
    
    now choosing C to be max(x), we ensure that the largest exponent becomes ZERO and all other exponents are in between 0 and 1. 
    '''
    row_max = tl.max(row, axis = 0)
    row_minus_max = row - row_max
    
    #@ Computing the exponentials
    numerator = tl.exp(row_minus_max)
    
    denominator = tl.sum(numerator, axis = 0)
    
    #@ Computing the softmax output
    softmax_output = numerator / denominator
    
    ##@ Writing back to DRAM:
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    
    tl.store(output_ptrs, softmax_output, mask = mask)

## Okay so the kernel is defined now defining a wraper function for easy use 

def softmax(x):
    ## Getting the dimensions 
    n_rows, n_cols = x.shape
    
    ## The block size is the next power of two greater than n_cols
    BLOCK_SIZE = triton.next_power_of_2(n_cols) ##@ wooo... some spicy function from trition a'ight
    
    ##@ Allocating the output 
    y= torch.empty_like(x)
    
    #@ Determining the warps based on BLOCK_SIZE
    num_warps = 4 #@ by default
    if BLOCK_SIZE > 2048:
        num_warps = 8
    elif BLOCK_SIZE > 4096:
        num_warps = 16
    
    ##@ Launching the kernel 
    grid = (n_rows,) #! Launching exactly one program for each row of the input matrix
    
    softmax_kernel[grid](
        y, x, 
        x.stride(0), y.stride(0), 
        n_rows, n_cols,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps
    ) 
    
    return y


if __name__ == "__main__":
    #@ Creating test data
    n_rows, n_cols = 3,5
    x= torch.randn((n_rows, n_cols), device = DEVICE)
    
    print(f"Input Tensor: {x}")
    
    ##@ Compute the softmax with triton
    y_triton = softmax(x)
    
    ##@ Computing with pyTorch for comparison
    y_torch = torch.nn.functional.softmax(x, dim = 1)
    
    print(f"\nTriton Softmax Output: {y_triton}")
    print(f"\nPyTorch Softmax Output: {y_torch}")
    
    #@ Comparing 
    diff = torch.max(torch.abs(y_triton - y_torch))
    print(f"Maximum Difference: {diff}")
    
    
    
    

    
    
    
    
    
    
    
    
    
