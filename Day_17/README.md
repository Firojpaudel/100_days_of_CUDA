## Summary of Day 17:

>* Starting from Parallel Convolution:

#### Parallel 2D Convolution in CUDA 

Convolution is highly parallelizable since each output element can be computed independently.

##### Trying to implement 2D Conv Kernel: _Inspired by Parallel Matrix Multiplication_

> To view the code implementation, [Click Here](./2D_convo.cu)
> Output of the code:
> ```shell
> Convolution result:
>0.444444 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.>666667 0.666667 0.444444 
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.666667 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.666667
>0.444444 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.666667 0.444444
>```
> Let's break down why the output looks like this:
>
>1. Convolution Operation:
> 
>     - Convolution involves sliding a kernel (a smaller matrix) over the input matrix and computing the dot product at each position.
>     - The result of each dot product forms the corresponding element in the output matrix.
>2. Edge Handling:
> 
>     - The values at the edges of the output matrix are typically lower because the kernel has fewer overlapping elements with the input matrix at the edges.
>     - This is evident from the lower values (0.444444 and 0.666667) at the corners and edges of the output.
> 3. Normalization:
>
>    - The values in the output matrix are normalized, which means they are scaled to a range, often between 0 and 1.
>    - This is why you see values like 0.444444 and 0.666667 instead of larger numbers.
>
> 4. Uniform Values in the Center:
>
>    - The center of the output matrix has uniform values (mostly 1) because the kernel fully overlaps with the input matrix in these regions, leading to consistent dot product results.
> 5. Kernel and Input Matrix:
>
>    - The specific values in the output matrix depend on the values in the input matrix and the kernel used for convolution.
>    - The pattern of values (0.444444, 0.666667, 1) suggests a smoothing or averaging effect, which is common in convolution operations.
>
> Here's a simplified example to illustrate the concept:
>```shell
>Input Matrix:
> 1 1 1
> 1 1 1
> 1 1 1
> 
> Kernel:
> 0.111 0.111 0.111
> 0.111 0.111 0.111
> 0.111 0.111 0.111
> 
> Output Matrix (after convolution):
> 0.444 0.666 0.444
> 0.666 1.000 0.666
> 0.444 0.666 0.444
> ```

#### Trying to Implement a Simple Image Convolution Kernel:

> _The code is in_ [image_conv](./image_conv.cu).


