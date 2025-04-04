## Summary of Day 67:

Okay so yesterday, we studied about Fourier Transform and FFT. Also, we were discussing that FFT is widely used in signal processing and image reconstruction. Today let's discuss on this.

#### Iterative Reconstruction

***Objective***: Reconstruct a voxel-based image $\rho$ from non-Cartesian k-space data $(D)$, using a quasi-Bayesian estimation framework.

>[!Note]
> Voxel-based image are $3D$ pixels that form a grid. They are particularly used in MRI and CT scans to create detailed $3D$ visualizations of scanned area. 
>
> quasi-Bayesian estimation is a flexible approach of Bayesian estimation that uses the prior knowledge to estimate the new one.

<details>
    <summary> <b>About Quasi-Bayesian Estimation </b></summary>
    <video width="640" height="360" controls>
    <source src="./images/Quasi-Bayesian (QB) learning - Benjamin Guedj (Spotlight session).mp4" type="video/mp4">
            Your browser does not support the video tag.
    </video>
</details>

***Problem Formulation***: 