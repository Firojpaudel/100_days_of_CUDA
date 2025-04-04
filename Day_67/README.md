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