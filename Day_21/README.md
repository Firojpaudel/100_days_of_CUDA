## Summary of Day 21:

> **Starting of new chapter; Chapter — 8 **Stencil***

#### Stencil Computation: _What is it?_
Stencil computations are a class of numerical algorithms used to **update elements in a data grid based on neighboring values**. They appear in many scientific and engineering applications, such as:

- **Solving Partial Differential Equations (PDEs)**
- **Heat Diffusion Simulations**
- **Image Processing _(Blurring, Edge Detection)_**
- **Fluid Dynamics & Weather Simulation**

Each point in a grid is updated by applying a fixed pattern (stencil) over its neighborhood.

---
#### Difference Between Stencil Computation and Convolution:
Though both apply a kernel to a region of data, they differ in intent:
|**Features**|	**Stencil Computation**|	**Convolution**|
|-------|----------------------|----------------|
|Purpose|	Used in PDEs, scientific computing|	Mostly used in deep learning & signal/image processing|
|Operation|	Updates each grid point based on its neighbors|	Computes weighted sum using a kernel|
|Data Dependency|	Strong dependencies on neighboring values|	Often applied with independent operations|

> _Stencil computations typically involve solving **differential equations**, while convolution is a **mathematical operation** used in filtering and feature extraction._

---
#### Stencils: 
> ***Definition:*** A stencil is a pattern of weights applied to grid points for numerical approximations of derivatives.

1. **Finite-Difference Approximation**
For a function $f(x)$, the first derivative is approximated as:

```math 
f'(x) = \frac{f(x+h)- f(x-h)}{2h} + O(h^2)
```
where:
- $h$: grid spacing
- $O(h^2)$: error term, proportional to $h^2$

> **Discrete Derivative Example:**
> Given grid array $F[i]$, 
> ```math
> F_{D}[i] = \frac{F[i+1]- F[i-1]}{2h}
> ```
> Rewriting as a weighted sum:
> ```math 
> F_{D}[i] = - \frac{1}{2h}F[i-1] + \frac{1}{2h} F[i+1]
> ```
> This defines a **three-point stencil**.