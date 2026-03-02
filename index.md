---
layout: default
title: 100 Days of CUDA
---

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

---

<div align="center">

<details open>
<summary><b>Days 01 - 20: CUDA Basics & Memory</b></summary>
<div style="overflow-x:auto;">
<table>
  <thead>
    <tr><th>Day</th><th>Learnt Topics</th><th>Links</th></tr>
  </thead>
  <tbody>
    <tr><td>Day 01</td><td>History, applications, setup, and first Hello World.</td><td><a href="./Day_01/">Day 01</a></td></tr>
    <tr><td>Day 02</td><td>Parameter passing, device queries, vector addition.</td><td><a href="./Day_02/">Day 02</a></td></tr>
    <tr><td>Day 03</td><td>Multidimensional grids, image color conversion.</td><td><a href="./Day_03/">Day 03</a></td></tr>
    <tr><td>Day 04</td><td>Image blurring, matrix multiplication.</td><td><a href="./Day_04/">Day 04</a></td></tr>
    <tr><td>Day 05</td><td>Modern GPU architecture, synchronization.</td><td><a href="./Day_05/">Day 05</a></td></tr>
    <tr><td>Day 06</td><td>Warps, SIMD, control divergence intro.</td><td><a href="./Day_06/">Day 06</a></td></tr>
    <tr><td>Day 07</td><td>Divergence impact on performance.</td><td><a href="./Day_07/">Day 07</a></td></tr>
    <tr><td>Day 08</td><td>Warp scheduling, occupancy.</td><td><a href="./Day_08/">Day 08</a></td></tr>
    <tr><td>Day 09</td><td>Memory access efficiency, roofline model.</td><td><a href="./Day_09/">Day 09</a></td></tr>
    <tr><td>Day 10</td><td>CUDA memory types: global, constant, shared.</td><td><a href="./Day_10/">Day 10</a></td></tr>
    <tr><td>Day 11</td><td>Tiling concept and memory tradeoffs.</td><td><a href="./Day_11/">Day 11</a></td></tr>
    <tr><td>Day 12</td><td>Tiled matrix multiplication, dynamic tiling.</td><td><a href="./Day_12/">Day 12</a></td></tr>
    <tr><td>Day 13</td><td>Memory coalescing, DRAM burst access.</td><td><a href="./Day_13/">Day 13</a></td></tr>
    <tr><td>Day 14</td><td>Corner turning, latency hiding.</td><td><a href="./Day_14/">Day 14</a></td></tr>
    <tr><td>Day 15</td><td>Thread coarsening, PMPP Ch 6 exercises.</td><td><a href="./Day_15/">Day 15</a></td></tr>
    <tr><td>Day 16</td><td>Convolutions: 1D and 2D intro.</td><td><a href="./Day_16/">Day 16</a></td></tr>
    <tr><td>Day 17</td><td>Parallel 2D convolution edge handling.</td><td><a href="./Day_17/">Day 17</a></td></tr>
    <tr><td>Day 18</td><td>2D convolution: pre/post-processing kernels.</td><td><a href="./Day_18/">Day 18</a></td></tr>
    <tr><td>Day 19</td><td>Constant memory, tiled convolution halo cells.</td><td><a href="./Day_19/">Day 19</a></td></tr>
    <tr><td>Day 20</td><td>Tiled convolution using caches for halo cells.</td><td><a href="./Day_20/">Day 20</a></td></tr>
  </tbody>
</table>
</div>
</details>

<details>
<summary><b>Days 21 - 40: Parallel Algorithms (Reduction, Scan, Merge)</b></summary>
<div style="overflow-x:auto;">
<table>
  <thead>
    <tr><th>Day</th><th>Learnt Topics</th><th>Links</th></tr>
  </thead>
  <tbody>
    <tr><td>Day 21</td><td>Stencil algorithms, parallel implementations.</td><td><a href="./Day_21/">Day 21</a></td></tr>
    <tr><td>Day 22</td><td>Optimization for 3D stencil computations.</td><td><a href="./Day_22/">Day 22</a></td></tr>
    <tr><td>Day 23</td><td>Exercises from Chapter 8.</td><td><a href="./Day_23/">Day 23</a></td></tr>
    <tr><td>Day 24</td><td>Parallel histogram implementation.</td><td><a href="./Day_24/">Day 24</a></td></tr>
    <tr><td>Day 25</td><td>Atomic operations, privatization, aggregation.</td><td><a href="./Day_25/">Day 25</a></td></tr>
    <tr><td>Day 26</td><td>Reduction: max and sum reduction.</td><td><a href="./Day_26/">Day 26</a></td></tr>
    <tr><td>Day 27</td><td>Convergent sum reduction kernel.</td><td><a href="./Day_27/">Day 27</a></td></tr>
    <tr><td>Day 28</td><td>Shared memory reduction, thread coarsening.</td><td><a href="./Day_28/">Day 28</a></td></tr>
    <tr><td>Day 29</td><td>Exercises from Chapter 10.</td><td><a href="./Day_29/">Day 29</a></td></tr>
    <tr><td>Day 30</td><td>Parallel prefix scan, Kogge-Stone intro.</td><td><a href="./Day_30/">Day 30</a></td></tr>
    <tr><td>Day 31</td><td>Kogge-Stone continued, complexity analysis.</td><td><a href="./Day_31/">Day 31</a></td></tr>
    <tr><td>Day 32</td><td>Brent-Kung parallel inclusive scan.</td><td><a href="./Day_32/">Day 32</a></td></tr>
    <tr><td>Day 33</td><td>Thread coarsening performance impact.</td><td><a href="./Day_33/">Day 33</a></td></tr>
    <tr><td>Day 34</td><td>Hierarchical scan algorithms.</td><td><a href="./Day_34/">Day 34</a></td></tr>
    <tr><td>Day 35</td><td>Exercises from Chapter 11.</td><td><a href="./Day_35/">Day 35</a></td></tr>
    <tr><td>Day 36</td><td>Parallel merge algorithms intro.</td><td><a href="./Day_36/">Day 36</a></td></tr>
    <tr><td>Day 37</td><td>Parallel merge kernels, co-ranks.</td><td><a href="./Day_37/">Day 37</a></td></tr>
    <tr><td>Day 38</td><td>Tiled merge kernels performance.</td><td><a href="./Day_38/">Day 38</a></td></tr>
    <tr><td>Day 39</td><td>Exercises from Chapter 12.</td><td><a href="./Day_39/">Day 39</a></td></tr>
    <tr><td>Day 40</td><td>Parallel radix sort implementation.</td><td><a href="./Day_40/">Day 40</a></td></tr>
  </tbody>
</table>
</div>
</details>

<details>
<summary><b>Days 41 - 60: Sparse Matrices, Graphs & ML Ops</b></summary>
<div style="overflow-x:auto;">
<table>
  <thead>
    <tr><th>Day</th><th>Learnt Topics</th><th>Links</th></tr>
  </thead>
  <tbody>
    <tr><td>Day 41</td><td>Multi-bit radix, parallel merge sort.</td><td><a href="./Day_41/">Day 41</a></td></tr>
    <tr><td>Day 42</td><td>Exercises from Chapter 13.</td><td><a href="./Day_42/">Day 42</a></td></tr>
    <tr><td>Day 43</td><td>SpMV with COO format implementation.</td><td><a href="./Day_43/">Day 43</a></td></tr>
    <tr><td>Day 44</td><td>CSR and ELL formats for sparse matrices.</td><td><a href="./Day_44/">Day 44</a></td></tr>
    <tr><td>Day 45</td><td>Hybrid ELL-COO and JDS formats.</td><td><a href="./Day_45/">Day 45</a></td></tr>
    <tr><td>Day 46</td><td>Exercises from Chapter 14.</td><td><a href="./Day_46/">Day 46</a></td></tr>
    <tr><td>Day 47</td><td>Normal BFS graph traversal.</td><td><a href="./Day_47/">Day 47</a></td></tr>
    <tr><td>Day 48</td><td>Vertex-centric parallelization: pull/push.</td><td><a href="./Day_48/">Day 48</a></td></tr>
    <tr><td>Day 49</td><td>Edge-centric and frontier-based processing.</td><td><a href="./Day_49/">Day 49</a></td></tr>
    <tr><td>Day 50</td><td>Exercises from Chapter 15.</td><td><a href="./Day_50/">Day 50</a></td></tr>
    <tr><td>Day 51</td><td>CNNs: basic ML concepts and architecture.</td><td><a href="./Day_51/">Day 51</a></td></tr>
    <tr><td>Day 52</td><td>PyCUDA: vector addition and matmul.</td><td><a href="./Day_52/">Day 52</a></td></tr>
    <tr><td>Day 53</td><td>CNN forward pass implementation.</td><td><a href="./Day_53/">Day 53</a></td></tr>
    <tr><td>Day 54</td><td>Backpropagation in CUDA intro.</td><td><a href="./Day_54/">Day 54</a></td></tr>
    <tr><td>Day 55</td><td>Complete backpropagation for CNN.</td><td><a href="./Day_55/">Day 55</a></td></tr>
    <tr><td>Day 56</td><td>ReLU activation function in PyCUDA.</td><td><a href="./Day_56/">Day 56</a></td></tr>
    <tr><td>Day 57</td><td>Matrix inversion kernel in PyCUDA.</td><td><a href="./Day_57/">Day 57</a></td></tr>
    <tr><td>Day 58</td><td>Batch normalization in PyCUDA.</td><td><a href="./Day_58/">Day 58</a></td></tr>
    <tr><td>Day 59</td><td>Layer normalization theory and code.</td><td><a href="./Day_59/">Day 59</a></td></tr>
    <tr><td>Day 60</td><td>Multi-Head Self-Attention in Triton intro.</td><td><a href="./Day_60/">Day 60</a></td></tr>
  </tbody>
</table>
</div>
</details>

<details>
<summary><b>Days 61 - 80: Triton, cuDNN, MRI & Competitions</b></summary>
<div style="overflow-x:auto;">
<table>
  <thead>
    <tr><th>Day</th><th>Learnt Topics</th><th>Links</th></tr>
  </thead>
  <tbody>
    <tr><td>Day 61</td><td>Fixed MHA Triton implementation details.</td><td><a href="./Day_61/">Day 61</a></td></tr>
    <tr><td>Day 62</td><td>CUDA CNN inference kernel design.</td><td><a href="./Day_62/">Day 62</a></td></tr>
    <tr><td>Day 63</td><td>cuDNN for DNN acceleration intro.</td><td><a href="./Day_63/">Day 63</a></td></tr>
    <tr><td>Day 64</td><td>Batch Norm with cuDNN implementation.</td><td><a href="./Day_64/">Day 64</a></td></tr>
    <tr><td>Day 65</td><td>Pooling forward pass (LeNet-5).</td><td><a href="./Day_65/">Day 65</a></td></tr>
    <tr><td>Day 66</td><td>MRI image reconstruction: k-space, FFT.</td><td><a href="./Day_66/">Day 66</a></td></tr>
    <tr><td>Day 67</td><td>Iterative MRI reconstruction challenges.</td><td><a href="./Day_67/">Day 67</a></td></tr>
    <tr><td>Day 68</td><td>F^H D kernel optimization for MRI.</td><td><a href="./Day_68/">Day 68</a></td></tr>
    <tr><td>Day 69</td><td>Dynamic Parallelism in CUDA, device launches.</td><td><a href="./Day_69/">Day 69</a></td></tr>
    <tr><td>Day 70</td><td>Tensara: Leaky ReLU and L1 Norm kernels.</td><td><a href="./Day_70/">Day 70</a></td></tr>
    <tr><td>Day 71</td><td>Tanh, Softmax, and Vector Addition unrolling.</td><td><a href="./Day_71/">Day 71</a></td></tr>
    <tr><td>Day 72</td><td>Matrix-Vector Multiplication performance.</td><td><a href="./Day_72/">Day 72</a></td></tr>
    <tr><td>Day 73</td><td>GEMM with bias and ReLU: C = ReLU(A.W+b).</td><td><a href="./Day_73/">Day 73</a></td></tr>
    <tr><td>Day 74</td><td>Inclusive Scan, Diagonal MatMul, ELU.</td><td><a href="./Day_74/">Day 74</a></td></tr>
    <tr><td>Day 75</td><td>Cumulative product: naive vs multi-kernel.</td><td><a href="./Day_75/">Day 75</a></td></tr>
    <tr><td>Day 76</td><td>Thrust for products, 4D/3D tensor matmul.</td><td><a href="./Day_76/">Day 76</a></td></tr>
    <tr><td>Day 77</td><td>Hinge Loss, SELU: Reached Rank 1 on Tensara.</td><td><a href="./Day_77/">Day 77</a></td></tr>
    <tr><td>Day 78</td><td>Swish activation: multiple approaches.</td><td><a href="./Day_78/">Day 78</a></td></tr>
    <tr><td>Day 79</td><td>RMS Normalization kernel benchmarking.</td><td><a href="./Day_79/">Day 79</a></td></tr>
    <tr><td>Day 80</td><td>Frobenius Norm and high GFLOPs MatMul.</td><td><a href="./Day_80/">Day 80</a></td></tr>
  </tbody>
</table>
</div>
</details>

<details>
<summary><b>Days 81 - 100: Advanced Optimization & Completion</b></summary>
<div style="overflow-x:auto;">
<table>
  <thead>
    <tr><th>Day</th><th>Learnt Topics</th><th>Links</th></tr>
  </thead>
  <tbody>
    <tr><td>Day 81</td><td>Frobenius Normalization implementation.</td><td><a href="./Day_81/">Day 81</a></td></tr>

    <tr><td>Day 82</td><td>Softplus and Min Over Dimension kernels.</td><td><a href="./Day_82/">Day 82</a></td></tr>
    <tr><td>Day 83</td><td>1D Convolution for Tensara competition.</td><td><a href="./Day_83/">Day 83</a></td></tr>
    <tr><td>Day 84</td><td>KL-Divergence kernel benchmarking.</td><td><a href="./Day_84/">Day 84</a></td></tr>
    <tr><td>Day 85</td><td>High GFLOPs Vector Addition and ReLU.</td><td><a href="./Day_85/">Day 85</a></td></tr>
    <tr><td>Day 86</td><td>Layer Normalization on 4D Tensors.</td><td><a href="./Day_86/">Day 86</a></td></tr>
    <tr><td>Day 87</td><td>Improved Leaky ReLU and Tri-MatMul.</td><td><a href="./Day_87/">Day 87</a></td></tr>
    <tr><td>Day 88</td><td>Upper Triangular Matrix Multiplication.</td><td><a href="./Day_88/">Day 88</a></td></tr>
    <tr><td>Day 89</td><td>L2 Normalization and optimized KL divergence.</td><td><a href="./Day_89/">Day 89</a></td></tr>
    <tr><td>Day 90</td><td>Symmetric MatMul and GEMM with bias/ReLU.</td><td><a href="./Day_90/">Day 90</a></td></tr>
    <tr><td>Day 91</td><td>Triplet Margin Loss optimization.</td><td><a href="./Day_91/">Day 91</a></td></tr>
    <tr><td>Day 92</td><td>GELU kernel and performance benchmarking.</td><td><a href="./Day_92/">Day 92</a></td></tr>
    <tr><td>Day 93</td><td>Product Over a Dimension kernel.</td><td><a href="./Day_93/">Day 93</a></td></tr>
    <tr><td>Day 94</td><td>2D Convolution: Naive vs Optimized.</td><td><a href="./Day_94/">Day 94</a></td></tr>
    <tr><td>Day 95</td><td>MSE Loss performance on H100/L40S.</td><td><a href="./Day_95/">Day 95</a></td></tr>
    <tr><td>Day 96</td><td>MSE Loss kernel analysis.</td><td><a href="./Day_96/">Day 96</a></td></tr>
    <tr><td>Day 97</td><td>Sigmoid Activation Function performance.</td><td><a href="./Day_97/">Day 97</a></td></tr>
    <tr><td>Day 98</td><td>MatMul with Swish and L1 Norm optimization.</td><td><a href="./Day_98/">Day 98</a></td></tr>
    <tr><td>Day 99</td><td>2D Average Pooling and MatMul with half2.</td><td><a href="./Day_99/">Day 99</a></td></tr>
    <tr><td>Day 100</td><td>2D Max Pooling and challenge completion.</td><td><a href="./Day_100/">Day 100</a></td></tr>
  </tbody>
</table>
</div>
</details>

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

<div align="center">
  <img src="https://www.svgrepo.com/show/373541/cuda.svg" alt="CUDA Icon" height="48"/>
  <img src="https://miro.medium.com/v2/resize:fit:1280/0*C4839mwCnQDzotdb" alt="Skill Icons" height="48"/>
  <br/>
  <img src="https://github-readme-stats.vercel.app/api/pin/?username=FirojPaudel&repo=100_days_of_CUDA&theme=radical" alt="GitHub Repo Card"/>
</div>
