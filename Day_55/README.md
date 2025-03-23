## Summary of Day 55:

Okay so yesterday, I focused on the $∂E/∂X$ calculation for the backpropagation of a convolutional layer in a CNN using CUDA C++. I derived the gradient of the loss $E$ with respect to the input $X$, which involved applying the chain rule and carefully handling the kernel offsets and bounds. Today, I’ll dive into the $∂E/∂W$ calculation, which is the gradient of the loss with respect to the weights $W$ of the convolutional layer. Let’s compare what’s different and build on yesterday’s understanding.

### Understanding $∂E/∂W$:

Just like yesterday, we’re working with a convolutional layer where the input $X$ has shape $[C, H_\text{in}, W_\text{in}]$, the output $Y$ has shape $[M, H_\text{out}, W_\text{out}]$, and the weights $W$ have shape $[M, C, K, K]$. The output dimensions, assuming stride of 1 and no padding, are still:

$$H_\text{out} = H_\text{in} - K + 1, \quad W_\text{out} = W_\text{in} - K + 1$$

The forward pass equation remains the same as yesterday:

$$Y[m, h_\text{out}, w_\text{out}] = \sum_{c=0}^{C-1} \sum_{p=0}^{K-1} \sum_{q=0}^{K-1} W[m, c, p, q] \cdot X[c, (h_\text{out} + p), (w_\text{out} + q)]$$

where $m$ is the output channel, $c$ is the input channel, and $(p, q)$ are the kernel offsets, with bound checking to ensure $h_\text{out} + p < H_\text{in}$ and $w_\text{out} + q < W_\text{in}$.

### Mathematical Derivation for $∂E/∂W$:

To compute $∂E/∂W$, we again use the chain rule, but this time we’re differentiating the loss $E$ with respect to the weights $W[m, c, p, q]$:

$$\frac{\partial E}{\partial W[m, c, p, q]} = \sum_{h_\text{out}} \sum_{w_\text{out}} \frac{\partial E}{\partial Y[m, h_\text{out}, w_\text{out}]} \cdot \frac{\partial Y[m, h_\text{out}, w_\text{out}]}{\partial W[m, c, p, q]}$$

#### Step 1: Compute $\frac{\partial Y[m, h_\text{out}, w_\text{out}]}{\partial W[m, c, p, q]}$:

Let’s look at the forward pass equation. The output $Y[m, h_\text{out}, w_\text{out}]$ depends on $W[m, c, p, q]$ directly. Differentiating the forward pass equation with respect to $W[m, c, p, q]$, we get:

$$\frac{\partial Y[m, h_\text{out}, w_\text{out}]}{\partial W[m, c, p, q]} = X[c, (h_\text{out} + p), (w_\text{out} + q)]$$

This makes sense because $W[m, c, p, q]$ is multiplied by $X[c, h_\text{out} + p, w_\text{out} + q]$ in the forward pass, and the derivative of a term like $W \cdot X$ with respect to $W$ is just $X$. Note that this term is non-zero only if the indices $h_\text{out} + p < H_\text{in}$ and $w_\text{out} + q < W_\text{in}$, which is already ensured by the bound checking in the forward pass.

#### Step 2: Apply the Chain Rule:

Now, substitute back into the chain rule expression:

$$\frac{\partial E}{\partial W[m, c, p, q]} = \sum_{h_\text{out}=0}^{H_\text{out}-1} \sum_{w_\text{out}=0}^{W_\text{out}-1} \frac{\partial E}{\partial Y[m, h_\text{out}, w_\text{out}]} \cdot X[c, (h_\text{out} + p), (w_\text{out} + q)]$$

Here, $\frac{\partial E}{\partial Y[m, h_\text{out}, w_\text{out}]}$ is the gradient of the loss with respect to the output $Y$, which is the $∂E/∂Y$ (or `dE_dY` in the code) that we already have from the backward pass of the next layer.

#### Step 3: Compare with $∂E/∂X$:

- **Similarity**: Both $∂E/∂X$ and $∂E/∂W$ use the chain rule and depend on $∂E/∂Y$. They both involve summing over certain dimensions to account for all contributions to the gradient.
- **Difference**: For $∂E/∂X$, we needed to figure out how $X[c, h, w]$ contributes to multiple $Y[m, h_\text{out}, w_\text{out}]$ values by sliding the kernel, which led to the flipped kernel $W[m, c, K-1-p, K-1-q]$. For $∂E/∂W$, the relationship is more direct: each $W[m, c, p, q]$ contributes to $Y[m, h_\text{out}, w_\text{out}]$ via a single $X[c, h_\text{out} + p, w_\text{out} + q]$, so there’s no kernel flipping involved.
- **Loop Structure**: In $∂E/∂X$, we looped over $h_\text{out}$ and $w_\text{out}$ in a constrained range based on $h$ and $w$ (e.g., $h_\text{out}$ from $h-K+1$ to $h$). In $∂E/∂W$, we loop over all $h_\text{out}$ and $w_\text{out}$ from 0 to $H_\text{out}-1$ and $W_\text{out}-1$, respectively, because each weight $W[m, c, p, q]$ affects all output positions.

#### Step 4: Relate to the Code:

Looking at the provided code for $∂E/∂W$, the implementation matches our derivation:

- The outer loops are over $m$, $c$, $p$, and $q$ to compute each $∂E/∂W[m, c, p, q]$.
- The inner loops are over $h_\text{out}$ (as $h$) and $w_\text{out}$ (as $w$), ranging from 0 to $H_\text{out}-1$ and $W_\text{out}-1$.
- The gradient is accumulated as $dE_dW[m, c, p, q] += dE_dY[m, h_\text{out}, w_\text{out}] \cdot X[c, h_\text{out} + p, w_\text{out} + q]$, exactly as derived.

This is different from the $∂E/∂X$ code (Figure 16.10), where the loops over $h_\text{out}$ and $w_\text{out}$ were constrained, and we used the flipped kernel $W[m, c, K-1-p, K-1-q]$.

### Final Expression:

The final expression for $∂E/∂W$ is:

$$\frac{\partial E}{\partial W[m, c, p, q]} = \sum_{h_\text{out}=0}^{H_\text{out}-1} \sum_{w_\text{out}=0}^{W_\text{out}-1} \frac{\partial E}{\partial Y[m, h_\text{out}, w_\text{out}]} \cdot X[c, h_\text{out} + p, w_\text{out} + q]$$

This gradient tells us how much each weight $W[m, c, p, q]$ contributes to the loss $E$, and we’ll use it to update the weights during backpropagation.


> **ⓘ Note:** \
> The $∂E/∂W$ calculation is more straightforward than $∂E/∂X$ because there’s no need to flip the kernel. However, it still requires careful indexing to ensure we’re accessing the correct $X$ values based on the kernel offsets $p$ and $q$. Tomorrow, I might explore how to optimize this CUDA implementation or move on to the next part of backpropagation, like handling biases or different strides/padding.

> [Click Here](./complete_backprop.cu) to redirect towards the complete backpropagation pipeline in CUDA for a convolution layer.