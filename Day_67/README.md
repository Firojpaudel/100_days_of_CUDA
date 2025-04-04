## Summary of Day 67:

Okay so yesterday, we studied about Fourier Transform and FFT. Also, we were discussing that FFT is widely used in signal processing and image reconstruction. Today let's discuss on this.

#### Iterative Reconstruction

***Objective***: Reconstruct a voxel-based image $\rho$ from non-Cartesian k-space data $(D)$, using a quasi-Bayesian estimation framework.

>[!Note]
> Voxel-based image are $3D$ pixels that form a grid. They are particularly used in MRI and CT scans to create detailed $3D$ visualizations of scanned area. 
>
> quasi-Bayesian estimation is a flexible approach of Bayesian estimation that uses the prior knowledge to estimate the new one.

<details>
  <summary><b>About Quasi-Bayesian Estimation</b></summary>
  <div align="center">
  <a href="https://www.youtube.com/watch?v=hncA61eBWDI" target="_blank">
    <img src="https://img.youtube.com/vi/hncA61eBWDI/0.jpg" alt="Watch the demo" style="max-width:100%; height:auto;">
  </a>
  </div>
</details>

***Problem Formulation***: 

$$(F^H F + \lambda W^H W)\rho = F^H D$$

where:
- $\rho$: Vector of voxel values for the reconstructed image
- $(F)$: Matrix modeling the physics of the imaging process *(eg, Fourier encoding)*
- $(D)$: Vector of k-space data samples from the scanner.
- $(W)$: Matrix incorporating prior information
- $\lambda$: Regularization parameter
- $F^H$: Conjugate transpose of $(F)$
- $W^H$: Conjugate transpose of $(W)$

> [!note]
> $F^H$ and $W^H$ are  also termed as Hermitian transposes on the book. 

***Challenges***:

While reconstructing an image from *k-space* data in medical imaging, such as **MRI** which rely on Fourier Transform we face major challenge of matrix sizes.

Here's the breakdown:
- For a modest $128 \times 128 \times 128$ voxel image, $(F)$ has $128^3$ ie $\approx 2 \text{ million columns}$, each column having $(N)$ elements. This leads to significant memory and computational challenges due to:
    - The high-dimensional nature of the Fourier transform, requiring efficient storage and computation.
    - The computational cost of applying FFT to large matrices, which scales as $O(N \log N)$, making real-time processing difficult.
    - Hardware constraints, as handling large FFT computations efficiently requires GPUs or specialized accelerators like TPUs.
>[!warning]
> Direct methods like Gaussian Elimination are impractical due to the massive dimensions of matrices

***Solution Approach***:
- We use an iterative method— Authors suggest to use **Conjugate Gradient (CG)** algorithm, to solve for $\rho$.

> [!note]
> When reconstructing an MRI image, we often solve system of equations of the form:
>
> $Ax= b$
>
> Where:
> - $A$ is the large symmetric positive definite matrix *(often derived from FT and system modeling)*
> - $x$ is the image we want to reconstruct
> - $b$ is the k-space data
>
> Since direct inversion of $A$ is infeasible due to its size, the **CG** provides an effecient way to iteratively *approximate* $x$ **without explicitly computing** $A^{-1}$.

> [!important]
> ***The Core Idea Behind CG***
>
> CG solves $Ax=b$ by searching for the solution in a sequence of conjugate directions, which helps converge in fewer iterations than a general iterative method *(like gradient descent)*
>
> Instead of solving for $x$ directly, **CG minimizes the quadratic function**:
>
> $f(x) = \frac{1}{2} x^T A x - b^T x$
>

> [!note]
>
> ---
> **Algorithm: Conjugate Gradient Method (CG)**
> 
> ---
> **Input:**
> - Symmetric Positive Definite (SPD) matrix $A \in \mathbb{R}^{n \times n}$
> - Right-hand side vector $b \in \mathbb{R}^{n}$
> - Initial guess $x_0$
> - Tolerance $\epsilon$
> ---
> 
> **Output:**
> - Approximate solution $x_k$ such that $||A x_k - b|| < \epsilon$
> ---
> 
> 1. **Initialize:**
>    $x_0, \quad r_0 = b - A x_0, \quad p_0 = r_0, \quad k = 0$
> 
> 2. **While** $||r_k|| > \epsilon$ **do:**
>    - Compute step size:
>      $\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$
>    - Update solution:
>      $x_{k+1} = x_k + \alpha_k p_k$
>    - Update residual:
>      $r_{k+1} = r_k - \alpha_k A p_k$
>    - Compute new search direction:
>      $\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$
>    - Update search direction:
>      $p_{k+1} = r_{k+1} + \beta_k p_k$
>    - Update index:
>      $k \gets k + 1$
> 
> 3. **End while**
> 
> 4. **Return** $x_k$
> ---

> [!important]
> - Unlike gradient descent, which moves along the steepest descent direction, CG constructs an optimal search direction by ensuring conjugacy, which accelerates convergence.
> <div align="center">
>   <img src="./images/imageA.png" width="400px">
>   <p><b>Fig 67_01: </b><i> An iterative linear solver based approach to reconstructing non-Cartesian k-space sample data. </i></p>
> </div>
>
> - For an $n$-dimensional problem, CG converges in at most $n$ iterations, much faster than gradient descent.

