## Summary of Day 73:

Trying to implement GEMM with bias and RELU activation.

$$C = \text{ReLU}(A \cdot W^T + b)$$

Where:

- $A \in \mathbb{R}^{B \times N}$: input matrix (batch size × input features)  
- $W \in \mathbb{R}^{M \times N}$: weight matrix  
- $b \in \mathbb{R}^{M} $: bias vector  
- $C \in \mathbb{R}^{B \times M}$: output after applying matrix multiplication, bias addition, and ReLU activation

> [Click Here](https://github.com/Firojpaudel/100_days_of_CUDA/blob/main/Day_73/GEMM_bias_relu.py) to see the code implementation
