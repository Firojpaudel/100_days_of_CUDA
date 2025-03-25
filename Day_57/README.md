## Summary of Day 57:

> Days until exams are over: $3$

Okay so today I thought of calculating inverse of a matrix using Triton 

> [Click Here](./inverse_matrix_kernel.py) to redirect towards code

> ***Output:***
> ```bash
> Sample Matrix [0]:
>  tensor([[ 2.3126,  2.0126],
>         [-0.3261, -0.3738]], device='cuda:0')
> Triton Inverse [0]:
>  tensor([[  1.7959,   9.6708],
>         [ -1.5668, -11.1122]], device='cuda:0')
> PyTorch Inverse [0]:
>  tensor([[  1.7959,   9.6708],
>         [ -1.5668, -11.1122]], device='cuda:0')
> Verification passed!
> 
> Triton Time: 0.0034 seconds
> PyTorch Time: 0.0038 seconds
> ```