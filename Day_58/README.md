## Summary of Day 58:

> Days until my exams are over: $2$

So, today I’m back at it, coding another kernel in Triton. So far, I’ve nailed:
- *Vector Addition*
- *Matrix Multiplication*
- *ReLU*
- *Inverse of a Matrix*

**Today’s Target**: Batch Normalization

## Batch Normalization: Theory and Triton Kernels

### How BatchNorm Works: Mathematical Derivation

Batch Normalization (BatchNorm) is a clutch move in deep learning—stabilizes training and speeds it up. It normalizes layer inputs across a batch to mean $0$ and variance $1$, then tweaks them with learnable params ($\gamma$ and $\beta$). Let’s break it down step-by-step for a batch shaped `[batch_size, features]`.

#### Step 1: Compute the Mean
For a batch $X= {x_{i,j}}$ where $i=1,\dots,\text{batch size}$ (samples) and $j=1,\dots,\text{features}$ (channels/features), compute the mean per feature $j$ across the batch:

$$
\mu_j = \frac{1}{\text{batch size}} \sum_{i=1}^{\text{batch size}} x_{i,j}
$$

- $\mu_j$: Mean of feature $j$ over all batch samples.
- Why? Centers data around $0$, cutting down internal covariate shift.

#### Step 2: Compute the Variance
Next, get the variance per feature $j$ across the batch, using that mean:

$$
\sigma_j^2 = \frac{1}{\text{batch size}} \sum_{i=1}^{\text{batch size}} (x_{i,j} - \mu_j)^2
$$

- $\sigma_j^2$: Variance of feature $j$, showing the spread.
- Note: Some use $\frac{1}{\text{batch size}-1}$ (unbiased), but training rolls with $\frac{1}{\text{batch size}}$.

#### Step 3: Normalize
Normalize each value with the mean and variance, tossing in a tiny $\epsilon$ to dodge division-by-zero drama:

```math
\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
```

- $\hat{x}_{i,j}$: Normalized value—mean ≈ $0$, variance ≈ $1$ (tweaked by $\epsilon$).
- $\epsilon$: Small constant (e.g., $1\times10^{-5}$) for stability.

#### Step 4: Scale and Shift
Hit it with learnable params $\gamma_j$ (scale) and $\beta_j$ (shift) per feature:

$$
y_{i,j} = \gamma_j \cdot \hat{x}_{i,j} + \beta_j
$$

- $y_{i,j}$: Final BatchNorm output.
- $\gamma_j,\beta_j$: Trained to adjust the normalized data’s range.

### Why It’s Dope
- Keeps gradients steady—activations don’t go wild.
- Speeds up training—no stress over initial weights.
- Everywhere in ML/DL/NLP: CNNs (ResNet), transformers (BERT), you name it.

---

### Triton Kernels: Breaking Down the Code

We’re splitting BatchNorm into two kernels: one for stats (mean/variance), one for normalization. Here’s the rundown with math and Triton vibes.

#### Kernel 1: `batchnorm_stats_kernel`
**Purpose**: Compute $\mu_{j}$ and $\sigma_{j^2}$ per feature across the batch.

**Math**:
- $\mu_j = \frac{1}{\text{batch size}} \sum_{i} x_{i,j}$
- $\sigma_j^2 = \frac{1}{\text{batch size}} \sum_{i} (x_{i,j} - \mu_j)^2$

**How It Works**:
- **Grid**: One block per feature ($\text{features}$ blocks).
- **Offsets**: For feature $j$, grab all batch values $x_{i,j}$ (stride-adjusted).
- **Compute**: 
  - Sum all $x_{i,j}$ and divide by $\text{batch size}$ for $\mu_j$.
  - Subtract $\mu_j$, square, sum, and divide for $\sigma_j^2$.
- **Store**: Write $\mu_j$ and $\sigma_j^2$ to output tensors.

**Triton Flex**: Parallelizes across features—each block owns a column.

#### Kernel 2: `batchnorm_norm_kernel`
**Purpose**: Normalize and scale/shift with precomputed stats.

**Math**:
- $y_{i,j} = \gamma_j \cdot \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}} + \beta_j$

**How It Works**:
- **Grid**: One block per batch item ($\text{batch size}$ blocks).
- **Offsets**: For batch item $i$, load all features $x_{i,j}$ (row-wise).
- **Load Stats**: Snag $\mu_j$, $\sigma_j^2$, $\gamma_j$, $\beta_j$ for all features.
- **Normalize**: Run the full formula per element.
- **Store**: Write $y_{i,j}$ to output.

**Triton Flex**: Parallelizes across batch items—each block tackles a row, vectorizing feature ops.

---

#### Why Two Kernels?
- **Efficiency**: Stats need a reduction (sum over batch), normalization’s pointwise. Splitting skips sync chaos.
- **Parallelism**: First kernel scales with $\text{features}$, second with $\text{batch size}$—maxes GPU threads.

> [Click Here](./batch_norm.py) to redirect towards the code implementation