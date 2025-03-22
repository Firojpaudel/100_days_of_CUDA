## Summary of Day 54:

Today, I'll try to learn the backpropagation in CNN using CUDA C++. 


> [Click Here](./backprop.cu) to redirect to today's code implementation.

Okay will just try to learn the $∂E/∂X$ calculation today. I want to get the maths right. 

So, before diving into the code section; let's understand what $∂E/∂X$ actually is.

In a CNN, The forward pass involves convolving the input $(X)$ of shape $[C, H_\text{in}, W_\text{in}]$ where $(C)$ is the number of input channels, $(H_\text{in})$ is the height and $(W_\text{in})$ is the width. And using these, we got output $(Y)$ of shape $[C, H_\text{out}, W_\text{out}]$. 

Now assuming the stride of $1$ and no padding, out output dimensions are: 
##### 
$$H_\text{out} = H_\text{in}−K+1,W_\text{out}=W_\text{in}−K+1$$

### A bit of Mathematical derivation: 

In forward pass, an output element $Y[m,h_\text{out},w_\text{out}]$ is computed as:
$$Y[m,h_\text{out},w_\text{out}] = \sum_{c=0}^{C-1}\sum_{p=0}^{K-1}\sum_{q=0}^{K-1}W[m,c,p,q] \cdot X[c, (h_\text{out}+ p), (w_\text{out} + q)]$$

**Where**:
- $m$ is the output channel;
- $c$ is the input channel, 
- and $(p,q)$ are the Kernel offsets. 

and in $X$ there's boundchecking just to ensure $h_\text{out}+ p < H_\text{in}$ and $w_\text{out} + q < W_\text{in}$. 

Now, to calculate the value of $∂E/∂X$, we could simply use chain rule as:

$$\frac{\partial E}{\partial X[c,h, w]} = \sum_m \sum_{h_\text{out}} \sum_{w_\text{out}} \frac{\partial E}{\partial Y[m,h_\text{out},w_\text{out}]} \cdot \frac{\partial Y[m,h_\text{out},w_\text{out}]}{\partial X[c,h, w]}$$ 

So, now we need to compute the value of $\frac{\partial Y[m,h_\text{out},w_\text{out}]}{\partial X[c,h, w]}$

- From the forward pass equation, we know that the $Y$ component depends on $X$ component only when the kernel position aligns such that $h^′+p=h$ and $w^′+q=w$

    > ***Note***: I'm writing as $h^\prime$ to denote $h_\text{out}$ just to make my typing easier. 
- So, **on solving we get:** $p = h - h^\prime$ and $q= w - w^\prime$

- And this is nonZero only if:
    - $0 \leq p < K$ and $0 \leq q < K$ i.e., $h - K + 1 \leq h^\prime \leq h$ and $w - K + 1 \leq w^\prime \leq w$ and,


    > ***Wondering how this came?***: Well, we had defined $H_\text{out}$ before in the initial equation and since $p$ is greater than or equals to $0$ from equation $0 \leq p < K$. we can infer $h - K + 1 \leq h^\prime \leq h$.


    - $h^\prime \geq 0, w^\prime \geq 0$ i.e., $h^\prime < H_\text{out}$ and $w^\prime < W_\text{out}$.

    - When these conditions hold; 
$$\frac{\partial Y[m, h', w']}{\partial X[c, h, w]} = W[m,c,p,q]
=W[m, c, h - h', w - w']$$

So **substituting and simplifying**:
$$\frac{\partial E}{\partial X[c, h, w]} = \sum_{m=0}^{M-1} \sum_{h' = h - K + 1}^{h} \sum_{w' = w - K + 1}^{w} \frac{\partial E}{\partial X[m, h^\prime, w^\prime]}\cdot W[m, c, h - h', w - w']$$

and $h^\prime$ ranges from $\text{max}(0, h-K+1)$ to $\text{min}(h, H_\text{out} -1)$ and $w^\prime$ ranges from $\text{max}(0, w-K+1)$ to $\text{min}(w, W_\text{out} -1)$.

Hence, leading to equation:
$$\frac{\partial E}{\partial X[c, h, w]} =
\sum_{m=0}^{M-1} \sum_{p=0}^{K-1} \sum_{q=0}^{K-1}
\frac{\partial E}{\partial Y[m, h-p, w-q]} \cdot W[m, c, K-1-p, K-1-q].$$

where $W[m, c, K-1-p, K-1-q]$ reflects the flipped kernel and we only include terms if $h-p$ and $w-q$ are within the bounds. 

