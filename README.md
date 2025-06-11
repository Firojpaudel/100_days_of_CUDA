# 100_days_of_CUDA
Challenging myself to learn CUDA (Basics ⇾ Intermediate) these 100 days. 

My learning resources: 
1. **Books**:
    - **Cuda By Example** _An Introduction to General-Purpose GPU Programming_ — Jason Sandres, Edward Kandrot
    - **PMPP**; _*4th Edition_ — Wen-mei, David, Izzat

<!-- Badges Section -->
<p align="center">
  <img src="https://img.shields.io/badge/Days_Completed-100-green?style=for-the-badge" alt="Days Completed"/>
  <img src="https://img.shields.io/badge/CUDA-Learning-blue?style=for-the-badge&logo=nvidia" alt="CUDA"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-Projects-orange?style=for-the-badge&logo=deepin" alt="Deep Learning"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">100 Days of CUDA 🚀</h1>

<p align="center" style="font-size:1.2em;">
  <b>Challenging myself to master CUDA programming, from basics to advanced deep learning, in 100 days.</b>
</p>

---

## 📚 Resources
- <b>Books:</b>
  - <b>Cuda By Example</b> — Jason Sandres, Edward Kandrot
  - <b>PMPP (4th Edition)</b> — Wen-mei, David, Izzat

---

<div align="center">

<table>
  <thead>
    <tr>
      <th>Day</th>
      <th>Learnt Topics</th>
      <th>Links</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Day 01</td>
      <td>History, applications, setup, and first Hello World CUDA program. Covers initial CUDA installation and running a basic kernel.</td>
      <td><a href="./Day_01/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 02</td>
      <td>Parameter passing, device queries, vector addition on kernel, and PMPP Chapter 2 exercises. Explores kernel arguments and device properties.</td>
      <td><a href="./Day_02/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 03</td>
      <td>Multidimensional grids, mapping threads to multidimensional data, and image color conversion. Practical thread mapping strategies.</td>
      <td><a href="./Day_03/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 04</td>
      <td>Image blurring, matrix multiplication, and solutions to exercises. Focus on convolution and matrix operations in CUDA.</td>
      <td><a href="./Day_04/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 05</td>
      <td>Modern GPU architecture, block scheduling, barrier synchronization, and use of __syncthreads().</td>
      <td><a href="./Day_05/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 06</td>
      <td>Warps, SIMD hardware, GPU architecture, and introduction to control divergence.</td>
      <td><a href="./Day_06/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 07</td>
      <td>Impact of divergence on performance, types of divergence, identification, and performance analysis.</td>
      <td><a href="./Day_07/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 08</td>
      <td>Warp scheduling, latency tolerance, resource partitioning, and occupancy.</td>
      <td><a href="./Day_08/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 09</td>
      <td>Memory access efficiency, roofline model, and matrix multiplication code optimization.</td>
      <td><a href="./Day_09/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 10</td>
      <td>CUDA memory types: global, constant, local, registers, and shared memory.</td>
      <td><a href="./Day_10/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 11</td>
      <td>Tiling concept and memory tradeoffs in CUDA matrix multiplication.</td>
      <td><a href="./Day_11/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 12</td>
      <td>Explanation for tiled matrix multiplication, impact of memory usage on occupancy, and dynamic tiling.</td>
      <td><a href="./Day_12/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 13</td>
      <td>Memory coalescing, row-major vs. column-major storage, and DRAM burst access in CUDA.</td>
      <td><a href="./Day_13/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 14</td>
      <td>Corner turning in matrix multiplication, memory coalescing analogies, and latency hiding.</td>
      <td><a href="./Day_14/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 15</td>
      <td>Thread coarsening and exercises from PMPP Chapter 6.</td>
      <td><a href="./Day_15/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 16</td>
      <td>Start of convolutions: 1D and 2D convolution with boundary conditions.</td>
      <td><a href="./Day_16/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 17</td>
      <td>Parallel 2D convolution with edge handling and normalization.</td>
      <td><a href="./Day_17/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 18</td>
      <td>Convolution on 2D images: preprocessing, CUDA kernel, and post-processing.</td>
      <td><a href="./Day_18/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 19</td>
      <td>Filter array properties, constant memory, caching, tiled convolution with halo cells, and thread strategies.</td>
      <td><a href="./Day_19/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 20</td>
      <td>Tiled convolution using caches for halo cells and exercises from Chapter 7.</td>
      <td><a href="./Day_20/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 21</td>
      <td>Stencil vs. convolution, parallel stencil algorithms, and code implementations.</td>
      <td><a href="./Day_21/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 22</td>
      <td>Thread coarsening and optimization for 3D stencil computations.</td>
      <td><a href="./Day_22/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 23</td>
      <td>Exercises from Chapter 8 and chapter completion.</td>
      <td><a href="./Day_23/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 24</td>
      <td>Introduction to parallel histogram and code implementation.</td>
      <td><a href="./Day_24/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 25</td>
      <td>Atomic operations, privatization, coarsening, and aggregation in CUDA.</td>
      <td><a href="./Day_25/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 26</td>
      <td>Reduction: max and sum reduction, and exercises from Chapter 10.</td>
      <td><a href="./Day_26/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 27</td>
      <td>Simple sum reduction kernel and convergent sum reduction.</td>
      <td><a href="./Day_27/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 28</td>
      <td>Shared memory for reduction, hierarchical reduction, and thread coarsening.</td>
      <td><a href="./Day_28/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 29</td>
      <td>Exercises from Chapter 10.</td>
      <td><a href="./Day_29/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 30</td>
      <td>Parallel prefix scan and Kogge-Stone algorithm.</td>
      <td><a href="./Day_30/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 31</td>
      <td>Kogge-Stone continued, complexity analysis, exclusive and inclusive scans.</td>
      <td><a href="./Day_31/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 32</td>
      <td>Brent-Kung parallel inclusive scan algorithm.</td>
      <td><a href="./Day_32/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 33</td>
      <td>Thread coarsening in detail and its impact on performance.</td>
      <td><a href="./Day_33/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 34</td>
      <td>Coarsening complexity analysis and hierarchical scan.</td>
      <td><a href="./Day_34/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 35</td>
      <td>Exercises from Chapter 11.</td>
      <td><a href="./Day_35/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 36</td>
      <td>Sequential merge and introduction to parallel merge algorithms.</td>
      <td><a href="./Day_36/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 37</td>
      <td>Parallel merge kernels, co-ranks, and divide and conquer strategies.</td>
      <td><a href="./Day_37/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 38</td>
      <td>Tiled merge kernels and their performance benefits.</td>
      <td><a href="./Day_38/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 39</td>
      <td>Exercises from Chapter 12.</td>
      <td><a href="./Day_39/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 40</td>
      <td>Parallel radix sort and its CUDA implementation.</td>
      <td><a href="./Day_40/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 41</td>
      <td>Choice of radix, multi-bit radix, memory coalescing, and parallel merge sort.</td>
      <td><a href="./Day_41/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 42</td>
      <td>Exercises from Chapter 13.</td>
      <td><a href="./Day_42/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 43</td>
      <td>SpMV with COO format and code implementation.</td>
      <td><a href="./Day_43/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 44</td>
      <td>CSR and ELL formats for sparse matrices in CUDA.</td>
      <td><a href="./Day_44/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 45</td>
      <td>Hybrid ELL-COO format, JDS format, and parallelization strategies.</td>
      <td><a href="./Day_45/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 46</td>
      <td>Exercises from Chapter 14.</td>
      <td><a href="./Day_46/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 47</td>
      <td>Normal BFS and introduction to graph traversal in CUDA.</td>
      <td><a href="./Day_47/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 48</td>
      <td>Vertex-centric parallelization: pull and push methods.</td>
      <td><a href="./Day_48/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 49</td>
      <td>Edge-centric parallelization and frontier-based graph processing.</td>
      <td><a href="./Day_49/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 50</td>
      <td>Privatization and exercises from Chapter 15.</td>
      <td><a href="./Day_50/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 51</td>
      <td>CNNs: basic ML concepts and CNN architecture.</td>
      <td><a href="./Day_51/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 52</td>
      <td>Vector addition and matrix multiplication in PyCUDA.</td>
      <td><a href="./Day_52/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 53</td>
      <td>CNN forward pass: CUDA implementation and performance.</td>
      <td><a href="./Day_53/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 54</td>
      <td>Backpropagation in CUDA: implementation and explanation.</td>
      <td><a href="./Day_54/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 55</td>
      <td>Complete backpropagation for CNN in CUDA.</td>
      <td><a href="./Day_55/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 56</td>
      <td>ReLU activation function in PyCUDA: implementation and testing.</td>
      <td><a href="./Day_56/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 57</td>
      <td>Matrix inversion kernel in PyCUDA and its applications.</td>
      <td><a href="./Day_57/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 58</td>
      <td>Batch normalization in PyCUDA: implementation and usage.</td>
      <td><a href="./Day_58/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 59</td>
      <td>Layer normalization in PyCUDA: theory and code.</td>
      <td><a href="./Day_59/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 60</td>
      <td>Multi-Head Self-Attention in Triton, initial implementation and notes.</td>
      <td><a href="./Day_60/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 61</td>
      <td>Fixed and explained MHA Triton implementation, detailed kernel parameter breakdown.</td>
      <td><a href="./Day_61/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 62</td>
      <td>CUDA CNN inference kernel design, thread organization, grid mapping.</td>
      <td><a href="./Day_62/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 63</td>
      <td>Explored cuDNN for DNN acceleration, convolution parameterization.</td>
      <td><a href="./Day_63/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 64</td>
      <td>Implemented Batch Norm with cuDNN, shared initial approach.</td>
      <td><a href="./Day_64/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 65</td>
      <td>Pooling forward pass (LeNet-5), memory layout discussion.</td>
      <td><a href="./Day_65/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 66</td>
      <td>MRI image reconstruction, k-space, FFT, and scan strategies.</td>
      <td><a href="./Day_66/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 67</td>
      <td>Iterative MRI reconstruction, quasi-Bayesian estimation, large matrix challenges.</td>
      <td><a href="./Day_67/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 68</td>
      <td>Step-by-step optimization of F^H D kernel for MRI, parallelization, atomic ops.</td>
      <td><a href="./Day_68/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 69</td>
      <td>Dynamic Parallelism in CUDA, device kernel launches, recursion.</td>
      <td><a href="./Day_69/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 70</td>
      <td>Tensara competition: Leaky ReLU and L1 Norm kernel submissions.</td>
      <td><a href="./Day_70/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 71</td>
      <td>Tanh, Softmax, and Vector Addition (loop unrolling, shared memory) kernels.</td>
      <td><a href="./Day_71/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 72</td>
      <td>Matrix Scalar Multiplication and Matrix Vector Multiplication, performance notes.</td>
      <td><a href="./Day_72/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 73</td>
      <td>GEMM with bias and ReLU activation: C = ReLU(A . W^T + b).</td>
      <td><a href="./Day_73/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 74</td>
      <td>Prefix Sum (Inclusive Scan), Diagonal Matrix Multiplication, ELU kernel.</td>
      <td><a href="./Day_74/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 75</td>
      <td>Cumulative product kernels: naive, multi-kernel, performance analysis.</td>
      <td><a href="./Day_75/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 76</td>
      <td>Fixed cumulative product with thrust, 4D/3D tensor matmul, cosine similarity.</td>
      <td><a href="./Day_76/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 77</td>
      <td>Hinge Loss, Hard Sigmoid, Huber Loss, SELU kernels; reached Tensara global rank 1.</td>
      <td><a href="./Day_77/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 78</td>
      <td>Swish activation function: multiple kernel approaches, benchmarking.</td>
      <td><a href="./Day_78/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 79</td>
      <td>RMS Normalization kernel and performance benchmarking.</td>
      <td><a href="./Day_79/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 80</td>
      <td>Optimized Frobenius Norm kernel and Mat-Mul kernel for high GFLOPs.</td>
      <td><a href="./Day_80/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 81</td>
      <td>Day missing or not documented.</td>
      <td></td>
    </tr>
    <tr>
      <td>Day 82</td>
      <td>Softplus kernel and Min Over Dimension kernel, performance notes.</td>
      <td><a href="./Day_82/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 83</td>
      <td>1D Convolution kernel for Tensara competition.</td>
      <td><a href="./Day_83/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 84</td>
      <td>KL-Divergence kernel and benchmarking on Tensara.</td>
      <td><a href="./Day_84/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 85</td>
      <td>Improved vector addition and ReLU kernel for higher GFLOPs.</td>
      <td><a href="./Day_85/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 86</td>
      <td>Layer Normalization kernel on 4D Tensor, performance benchmarking.</td>
      <td><a href="./Day_86/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 87</td>
      <td>Improved Leaky ReLU and Lower Triangular Matrix Multiplication kernels.</td>
      <td><a href="./Day_87/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 88</td>
      <td>Upper Triangular Matrix Multiplication kernel, performance notes.</td>
      <td><a href="./Day_88/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 89</td>
      <td>L2 Normalization and optimized KL divergence kernels.</td>
      <td><a href="./Day_89/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 90</td>
      <td>Symmetric Matrix Multiplication and GEMM with bias and ReLU kernels.</td>
      <td><a href="./Day_90/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 91</td>
      <td>Triplet Margin Loss and optimized Softplus kernel.</td>
      <td><a href="./Day_91/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 92</td>
      <td>GELU kernel and performance benchmarking.</td>
      <td><a href="./Day_92/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 93</td>
      <td>Product Over a Dimension kernel: two implementations, performance notes.</td>
      <td><a href="./Day_93/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 94</td>
      <td>2D Convolution kernel: naive, optimized, performance comparison.</td>
      <td><a href="./Day_94/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 95</td>
      <td>MSE Loss kernel and performance on H100 and L40S GPUs.</td>
      <td><a href="./Day_95/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 96</td>
      <td>MSE Loss kernel and performance on H100 and L40S GPUs.</td>
      <td><a href="./Day_96/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 97</td>
      <td>Sigmoid Activation Function kernel and performance notes.</td>
      <td><a href="./Day_97/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 98</td>
      <td>Matrix Multiplication with Swish Activation and optimized L1 Norm kernel.</td>
      <td><a href="./Day_98/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 99</td>
      <td>2D Average Pooling and optimized MatMul kernel with half2 and __hfma2.</td>
      <td><a href="./Day_99/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 100</td>
      <td>2D Max Pooling kernel and challenge completion reflection.</td>
      <td><a href="./Day_100/README.md">README</a></td>
    </tr>
  </tbody>
</table>

</div>

---

## ✨ Project Highlights
- **Comprehensive Coverage:** From CUDA basics to advanced deep learning and transformer architectures.
- **Hands-on Code:** Every day features real CUDA code, with a focus on practical, high-performance GPU programming.
- **Modern Deep Learning:** Includes CNNs, RNNs, attention mechanisms, normalization, and more.
- **Performance Optimization:** Profiling, memory management, and multi-GPU strategies.

---

## 🌟 What's Next?
<div align="center">
  <b>Stay tuned for more advanced CUDA explorations, real-world projects, and deep dives into GPU-powered AI!</b><br/>
  <sub>Follow this repository for future updates and bonus content.</sub>
</div>

---

## 📊 Visual Summary
<div align="center">
  <img src=https://www.svgrepo.com/show/373541/cuda.svg alt="Skill Icons" height="48"/>
  <img src="https://miro.medium.com/v2/resize:fit:1280/0*C4839mwCnQDzotdb" alt="Skill Icons" height="48"/>
  <br/>
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=FirojPaudel&repo=100_days_of_CUDA&theme=radical" alt="GitHub Repo Card"/>
</div>

---

