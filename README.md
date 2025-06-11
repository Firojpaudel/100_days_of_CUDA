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
      <td>History, Applications, Setup, First Hello World CUDA program</td>
      <td><a href="./Day_01/README.md">README</a>, <a href="./Day_01/hello.cu">Code</a></td>
    </tr>
    <tr>
      <td>Day 02</td>
      <td>Parameters Passing, Device Queries, Vector Addition on Kernel, PMPP Chapter_02 Exercises Solved</td>
      <td><a href="./Day_02/README.md">README</a>, <a href="./Day_02/params.cu">Params_passing Code</a>, <a href="./Day_02/dev_queries.cu">Dev_Query Code</a>, <a href="./Day_02/vect_addn.cu">Vect_addn Code</a></td>
    </tr>
    <tr>
      <td>Day 03</td>
      <td>Multidimensional Grids Organization, Mapping Threads to Multidimensional Data</td>
      <td><a href="./Day_03/README.md">README</a>, <a href="./Day_03/grids.cu">Code for Grids Explanation</a>, <a href="./Day_03/image_color_conv.cu">RGB to Grayscale Conversion Code</a></td>
    </tr>
    <tr>
      <td>Day 04</td>
      <td>Image Blurring, Matrix Multiplication, Solution to Exercise 1(a,b) and 2</td>
      <td><a href="./Day_04/README.md">README</a>, <a href="./Day_04/image_blur.cu">Image Blurring Code</a>, <a href="./Day_04/Exercise_01_soln_a.cu">Exercise_Qn.1(a) Code</a>, <a href="./Day_04/Exercise_01_soln_b.cu">Exercise_Qn.1(b) Code</a>, <a href="./Day_04/Exercise_02_soln.cu">Exercise 2 Code</a></td>
    </tr>
    <tr>
      <td>Day 05</td>
      <td>Architecture of modern GPU _(intro)_, Architecture diagram understanding, Block Scheduling, Barrier Synchronization and using `__syncthreads()`</td>
      <td><a href="./Day_05/README.md">README</a>, <a href="./Day_05/barrier_sync.cu">Barrier Synchronization Code</a></td>
    </tr>
    <tr>
      <td>Day 06</td>
      <td>Warps and SIMD Hardware, The modern GPU architecture, Control Divergence _Intro_</td>
      <td><a href="./Day_06/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 07</td>
      <td>Studying impacts of divergence on Performance, Types of divergence _(If-else & Loop-Based)_, Divergence identification, Performance Impace Analysis</td>
      <td><a href="./Day_07/README.md">README</a>, <a href="./Day_07/if-else_diverge.cu">if_else divergence Code</a>, <a href="./Day_07/loops_warp_divergence.cu">loops_warp_divergence Code</a></td>
    </tr>
    <tr>
      <td>Day 08</td>
      <td>Warp Scheduling and Latency Tolerance, Resource Partitioning & Occupancy</td>
      <td><a href="./Day_08/README.md">README</a>, <a href="./Day_08/Exercise_02.cu">Exercise_02_Solution Code</a>, <a href="./Day_08/Exercise_04.cu">Exercise_04_Solution Code</a></td>
    </tr>
    <tr>
      <td>Day 09</td>
      <td>Memory Access Effeciency in CUDA, Roofline Model, Matrix Multiplication Code Optimization</td>
      <td><a href="./Day_09/README.md">README</a>, <a href="./Day_09/matrix_multiplication.cu">Matrix Multiplication Code</a>, <a href="./Day_09/optimized_mat_mul.cu">Optimized Matrix Multiplication Code</a></td>
    </tr>
    <tr>
      <td>Day 10</td>
      <td>CUDA Memory Types: Global, Constant, Local, Registers, Shared</td>
      <td><a href="./Day_10/README.md">README</a>, <a href="./Day_10/mem_types_in_action.cu">Memory Types Code</a></td>
    </tr>
    <tr>
      <td>Day 11</td>
      <td>Tiling Concept and Memory Tradeoffs</td>
      <td><a href="./Day_11/README.md">README</a>, <a href="./Day_11/tiled_mat_mul.cu">Tiled Matrix Multiplication Code</a></td>
    </tr>
    <tr>
      <td>Day 12</td>
      <td>Explanation for Day 11 Tiled Matrix Multiplication Code, Impact of Memory Usage on Occupancy</td>
      <td><a href="./Day_12/README.md">README</a>, <a href="./Day_12/Day_12_updated_code.cu">Dynamic Tiled Matrix Multiplication Kernel Code</a></td>
    </tr>
    <tr>
      <td>Day 13</td>
      <td>Memory Coalescing in CUDA, Row-Major vs. Column-Major Storage, Coalsced Memory Access in CUDA, Understanding DRAM and Burst Access</td>
      <td><a href="./Day_13/README.md">README</a>, <a href="./Day_13/row_vs_column_major.cu"> Row VS Column Majors Code</a></td>
    </tr>
    <tr>
      <td>Day 14</td>
      <td>Explaining the Corner Turning in Mat Mul, Memory Coalescing with a bit of Analogy, Memory Latency Hiding</td>
      <td><a href="./Day_14/README.md">README</a>, <a href="./Day_14/corner_turning.cu">Code for Corner Turning</a></td>
    </tr>
    <tr>
      <td>Day 15</td>
      <td>Thread Coarsening, Exercises of Chapter 6 of PMPP</td>
      <td><a href="./Day_15/README.md">README</a>, <a href="./Day_15/thread_coarsening.cu">Code For Thread Coarsening</a></td>
    </tr>
    <tr>
      <td>Day 16</td>
      <td>Start of Chapter 7: Convolutions; 1D and 2D Convolution with Boundary Conditions</td>
      <td><a href="./Day_16/README.md">README</a>, <a href="./Day_16/1D_Conv.cu"> Code For 1D Convolution</a>, <a href="./Day_16/2D_Conv.cu">Code For 2D Convolution</a></td>
    </tr>
    <tr>
      <td>Day 17</td>
      <td>Parallel 2D Convolution implementation with Edge Handling, Normalization</td>
      <td><a href="./Day_17/README.md">README</a>, <a href="./Day_17/2D_convo.cu">Code For 2D Convolution with Edge Handlings</a></td>
    </tr>
    <tr>
      <td>Day 18</td>
      <td>Implementation of Convolution on 2D image</td>
      <td><a href="./Day_18/README.md">README</a>, <a href="./Day_18/Convolution/prepare.py">Image Preprocessing Code</a>, <a href="./Day_18/Convolution/Convolution_img.cu">CUDA Convolution Kernel Code</a>, <a href="./Day_18/Convolution/post_processing.py">Code For Post_Processing and Displaying</a></td>
    </tr>
    <tr>
      <td>Day 19</td>
      <td>Properties of Filter Array in Conv, Constant memory in CUDA, Caching in CUDA and Memory Hierarchy, Tiled Convolution with Halo Cells, Thread Organization Strategies</td>
      <td><a href="./Day_19/README.md">README</a>, <a href="./Day_19/tiled_2D_conv.cu">Tiled 2D convolution Code</a></td>
    </tr>
    <tr>
      <td>Day 20</td>
      <td> Tiled Convolution Using Caches for Halo Cells , Exercises from Chapter 7</td>
      <td><a href="./Day_20/README.md">README</a>, <a href="./Day_20/tiled_2D_with_cache.cu">Tiled 2D Conv Code</a>, <a href="./Day_20/3D.cu">3D Convolution basic kernel</a></td>
    </tr>
    <tr>
      <td>Day 21</td>
      <td>Chapter 8: **Stencil**, Differenece between Stencil and Convolution, Parallel Stencil (Algos)</td>
      <td><a href="./Day_21/README.md">README</a>, <a href="./Day_21/basic_stencil.cu">Basic Stencil Code</a>, <a href="./Day_21/optimized_stencil.cu">Optimized Stencil Code</a></td>
    </tr>
    <tr>
      <td>Day 22</td>
      <td>Thread Coarsening and optimizing 3D Stencil computations through it, Thread Coarsening Architecture</td>
      <td><a href="./Day_22/README.md">README</a> , <a href="./Day_22/thread_coarsening1.cu">3D Stencil: Unoptimized Code</a>, <a href="./Day_22/thread_coarsening2.cu">3D Stencil: Optimized Code</a></td>
    </tr>
    <tr>
      <td>Day 23</td>
      <td>Exercises from Chapter 8; Chapter Completion</td>
      <td><a href="./Day_23/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 24</td>
      <td>Starting Chapter 9: **Parallel Histogram** Introudction with code</td>
      <td><a href="./Day_24/parallel_hist.cu">Parallel Historam Code</a></td>
    </tr>
    <tr>
      <td>Day 25</td>
      <td>Atomic Ops</td>
      <td><a href="./Day_25/README.md">README</a>, <a href="./Day_25/privatization.cu">Privatization Code</a>, <a href="./Day_25/coarserning.cu">Coarsening Code</a>, <a href="./Day_25/aggregation.cu">Aggregation Code</a></td>
    </tr>
    <tr>
      <td>Day 26</td>
      <td>Ending of Chapter 9: Exercises, Chapter 10: **Reduction** Start</td>
      <td><a href="./Day_26/README.md">README</a>, <a href="./Day_26/max_reduction.cu">Max Reduction Code</a>, <a href="./Day_26/sum_reduction.cu">Sum Reduction Code</a></td>
    </tr>
    <tr>
      <td>Day 27</td>
      <td>Simple Sum Reduction Kernel, Convergent Sum Reduction</td>
      <td><a href="./Day_27/README.md">README</a>, <a href="./day_27/SimpleSumReductionKernel.cu">Simple Sum Reduction Code</a>, <a href="./day_27/optimizedKernel.cu">Convergent Sum Reduction Code</a></td>
    </tr>
    <tr>
      <td>Day 28</td>
      <td>Shared Memory For Reduction, Hierarchial Reduction, Thread Coarsening for Reduced Overheads</td>
      <td><a href="./Day_28/README.md">README</a>, <a href="./Day_28/shared_mem.cu">Shared Memory Reduction Code</a>, <a href="./Day_28/hierarchial.cu">Hierarchial Reduction Code</a>, <a href="./Day_28/Coarsened.cu">Coarsening Reduction Code</a></td>
    </tr>
    <tr>
      <td>Day 29</td>
      <td>Exercises from Chapter 10</td>
      <td><a href="./Day_29/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 30</td>
      <td>Parallel Prefix Scan, Kogge-Stone Parallel Prefix Scan Algo</td>
      <td><a href="./Day_30/README.md">README</a>, <a href="./Day_30/koggle_stone.cu">Kogge Stone Code</a></td>
    </tr>
    <tr>
      <td>Day 31</td>
      <td>Kogge Stone Continue, Complexity Analysis (Both Exclusive and Inclusive)</td>
      <td><a href="./Day_31/README.md">README</a>, <a href="./Day_31/exclusive_scan.cu">Exclusive Scan Code</a>, <a href="./Day_31/inclusive_scan.cu">Inclusive Scan Code</a></td>
    </tr>
    <tr>
      <td>Day 32</td>
      <td>Brent- Kung Parallel Inclusive Scan Algo</td>
      <td><a href="./Day_32/README.md">README</a>, <a href="./Day_32/Brent_kung.cu">Brent Kung Code</a></td>
    </tr>
    <tr>
      <td>Day 33</td>
      <td>Coarsening in Detail</td>
      <td><a href="./Day_33/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 34</td>
      <td>Coarsening Complexity Analysis, Hierarchial Scan</td>
      <td><a href="./Day_34/README.md">README</a>, <a href="./Day_34/coarsening.cu">Coarsening Code</a>, <a href="./Day_34/hierarchial.cu">Hierarchial Scan Code</a></td>
    </tr>
    <tr>
      <td>Day 35</td>
      <td>Exercises from Chapter 11</td>
      <td><a href="./Day_35/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 36</td>
      <td>Chapter 12: Merge, Sequential Merge</td>
      <td><a href="./Day_36/README.md">README</a>, <a href="./Day_36/seq_merge.cu">Sequential Merge Code</a></td>
    </tr>
    <tr>
      <td>Day 37</td>
      <td>Parallel Merge Kernels and Co-ranks</td>
      <td><a href="./Day_37/README.md">README</a>, <a href="./Day_37/corank.cu">CoRank Implementation Code</a>, <a href="./Day_37/div_conq.cu">Divide and Conquer Code</a></td>
    </tr>
    <tr>
      <td>Day 38</td>
      <td>Tiled Merge Kernels</td>
      <td><a href="./Day_38/README.md">README</a>, <a href="./Day_38/tiled_merged.cu">Tiled Merge Code</a></td>
    </tr>
    <tr>
      <td>Day 39</td>
      <td>Exercises from Chapter 12</td>
      <td><a href="./Day_39/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 40</td>
      <td>Paralel Radix Sort</td>
      <td><a href="./Day_40/README.md">README</a>, <a href="./Day_40/parallel_radix.cu">Radix Sort Code</a></td>
    </tr>
    <tr>
      <td>Day 41</td>
      <td>Choice of Radix, Multi-bit radix, Optimizing Memory Coalescening Using Parallel Radix Sort, Thread Coarsening To Improve Memory Coalescening, Parallel Merge Sort</td>
      <td><a href="./Day_41/README.md">README</a>, <a href="./Day_41/coarsening_applied.cu">Coarsening Applied Code</a>, <a href="./Day_41/parallel_merge_sort.cu">Parallel Merge</a></td>
    </tr>
    <tr>
      <td>Day 42</td>
      <td>Exercises from Chapter 13</td>
      <td><a href="./Day_42/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 43</td>
      <td>SmPV with COO</td>
      <td><a href="./Day_43/README.md">README</a>, <a href="./Day_43/SpMV_COO.cu">SmPV COO Code Implementation</a></td>
    </tr>
    <tr>
      <td>Day 44</td>
      <td>CSR Format, ELL Format</td>
      <td><a href="./Day_44/README.md">README</a>, <a href="./Day_44/CSR_Implementation.cu">CSR Implementation Code</a></td>
    </tr>
    <tr>
      <td>Day 45</td>
      <td>Hybrid ELL-COO Format, JDS Format with parallelization</td>
      <td><a href="./Day_45/README.md">README</a>, <a href="./Day_45/ELL_COO.cu">Hybrid ELL-COO Code</a></td>
    </tr>
    <tr>
      <td>Day 46</td>
      <td>Exercises from Chapter 14</td>
      <td><a href="./Day_46/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 47</td>
      <td>Chapter 15 Starts, Normal BFS</td>
      <td><a href="./Day_47/README.md">README</a>, <a href="./Day_47/bfs_simple.cu">Normal BFS Code</a></td>
    </tr>
    <tr>
      <td>Day 48</td>
      <td>Vertex Centric Parallelization, Vertex Centric Pull; Push</td>
      <td><a href="./Day_48/README.md">README</a>, <a href="./Day_48/vertex_centric_pull.cu">Vertex Centric Code</a>, <a href="./Day_48/vertex_centric_push.cu">Vertex Centric Push Code</a></td>
    </tr>
    <tr>
      <td>Day 49</td>
      <td>Edge Centric Pararallelization, Frontiers</td>
      <td><a href="./Day_49/README.md">README</a>, <a href="./Day_49/edge_centric.cu">Edge Centric Code</a>, <a href="./Day_49/frontier.cu">Frontier Code</a></td>
    </tr>
    <tr>
      <td>Day 50</td>
      <td>Privatization, Exercises from Chapter 15</td>
      <td><a href="./Day_50/README.md">README</a> , <a href="./Day_50/privatization.cu">Privatization Code</a></td>
    </tr>
    <tr>
      <td>Day 51</td>
      <td>CNNs: Basic ML Concepts, CNN Architecture</td>
      <td><a href="./Day_51/README.md">README</a>, <a href="./Day_51/CNN_implementation.cu">CNN Implementation Code</a></td>
    </tr>
    <tr>
      <td>Day 52</td>
      <td>Vector Addition and Matrix Multiplication in PyCUDA</td>
      <td><a href="./Day_52/README.md">README</a>, <a href="./Day_52/vect_add.py">Vector Add Code</a>, <a href="./Day_52/mat_mul.py">Matrix Mul Code</a></td>
    </tr>
    <tr>
      <td>Day 53</td>
      <td>CNN Forward Pass: CUDA Implementation</td>
      <td><a href="./Day_53/README.md">README</a>, <a href="./Day_51/CNN_implementation.cu">CNN CUDA Code</a></td>
    </tr>
    <tr>
      <td>Day 54</td>
      <td>Backpropagation: CUDA Implementation</td>
      <td><a href="./Day_54/README.md">README</a>, <a href="./Day_54/backprop.cu">Backprop Code</a></td>
    </tr>
    <tr>
      <td>Day 55</td>
      <td>Complete Backpropagation for CNN in CUDA</td>
      <td><a href="./Day_55/README.md">README</a>, <a href="./Day_55/complete_backprop.cu">Complete Backprop Code</a></td>
    </tr>
    <tr>
      <td>Day 56</td>
      <td>ReLU Activation Function in PyCUDA</td>
      <td><a href="./Day_56/README.md">README</a>, <a href="./Day_56/relu.py">ReLU Code</a></td>
    </tr>
    <tr>
      <td>Day 57</td>
      <td>Matrix Inversion Kernel in PyCUDA</td>
      <td><a href="./Day_57/README.md">README</a>, <a href="./Day_57/inverse_matrix_kernel.py">Inverse Matrix Kernel</a></td>
    </tr>
    <tr>
      <td>Day 58</td>
      <td>Batch Normalization in PyCUDA</td>
      <td><a href="./Day_58/README.md">README</a>, <a href="./Day_58/batch_norm.py">BatchNorm Code</a></td>
    </tr>
    <tr>
      <td>Day 59</td>
      <td>Layer Normalization in PyCUDA</td>
      <td><a href="./Day_59/README.md">README</a>, <a href="./Day_59/layerNorm.py">LayerNorm Code</a></td>
    </tr>
    <tr>
      <td>Day 60</td>
      <td>Multi-Head Attention (MHA) in PyCUDA</td>
      <td><a href="./Day_60/README.md">README</a>, <a href="./Day_60/MHA.py">MHA Code</a></td>
    </tr>
    <tr>
      <td>Day 61</td>
      <td>Transformer Layer: Theory and Implementation</td>
      <td><a href="./Day_61/README.md">README</a></td>
    </tr>
    <tr>
      <td>Day 62</td>
      <td>GEMM-based Convolution in CUDA</td>
      <td><a href="./Day_62/README.md">README</a>, <a href="./Day_62/Conv_GEMM.cu">Conv GEMM Code</a></td>
    </tr>
    <tr>
      <td>Day 63</td>
      <td>cuDNN Convolution Integration</td>
      <td><a href="./Day_63/README.md">README</a>, <a href="./Day_63/cuDNN_convolution.cu">cuDNN Convolution Code</a></td>
    </tr>
    <tr>
      <td>Day 64</td>
      <td>Training Loop and Loss Calculation in CUDA</td>
      <td><a href="./Day_64/README.md">README</a>, <a href="./Day_64/training_loop.cu">Training Loop Code</a></td>
    </tr>
    <tr>
      <td>Day 65</td>
      <td>Model Evaluation and Accuracy Metrics</td>
      <td><a href="./Day_65/README.md">README</a>, <a href="./Day_65/evaluation.cu">Evaluation Code</a></td>
    </tr>
    <tr>
      <td>Day 66</td>
      <td>Fast Fourier Transform (FFT) in CUDA</td>
      <td><a href="./Day_66/README.md">README</a>, <a href="./Day_66/fft.cu">FFT Code</a></td>
    </tr>
    <tr>
      <td>Day 67</td>
      <td>Fourier Domain Convolution for CNNs</td>
      <td><a href="./Day_67/README.md">README</a>, <a href="./Day_67/fourier_conv.cu">Fourier Conv Code</a></td>
    </tr>
    <tr>
      <td>Day 68</td>
      <td>Memory Access Optimization in CUDA</td>
      <td><a href="./Day_68/README.md">README</a>, <a href="./Day_68/memory_access.cu">Memory Access Code</a></td>
    </tr>
    <tr>
      <td>Day 69</td>
      <td>Dynamic Parallelism in Deep Learning</td>
      <td><a href="./Day_69/README.md">README</a>, <a href="./Day_69/dynamic_parallel.cu">Dynamic Parallelism Code</a></td>
    </tr>
    <tr>
      <td>Day 70</td>
      <td>Optimized Matrix Multiplication for FC Layers</td>
      <td><a href="./Day_70/README.md">README</a>, <a href="./Day_70/optimized_gemm.cu">Optimized GEMM Code</a></td>
    </tr>
    <tr>
      <td>Day 71</td>
      <td>Thread Coarsening in Neural Networks</td>
      <td><a href="./Day_71/README.md">README</a>, <a href="./Day_71/thread_coarsening.cu">Thread Coarsening Code</a></td>
    </tr>
    <tr>
      <td>Day 72</td>
      <td>Dropout Layer Implementation in CUDA</td>
      <td><a href="./Day_72/README.md">README</a>, <a href="./Day_72/dropout.cu">Dropout Code</a></td>
    </tr>
    <tr>
      <td>Day 73</td>
      <td>Custom CUDA Kernels for Activation Functions</td>
      <td><a href="./Day_73/README.md">README</a>, <a href="./Day_73/activation_functions.cu">Activation Functions Code</a></td>
    </tr>
    <tr>
      <td>Day 74</td>
      <td>Im2Col Optimization for Convolution</td>
      <td><a href="./Day_74/README.md">README</a>, <a href="./Day_74/im2col.cu">Im2Col Code</a></td>
    </tr>
    <tr>
      <td>Day 75</td>
      <td>Skip Connections and Residual Blocks</td>
      <td><a href="./Day_75/README.md">README</a>, <a href="./Day_75/resnet.cu">ResNet Code</a></td>
    </tr>
    <tr>
      <td>Day 76</td>
      <td>Memory-Efficient Feature Map Management</td>
      <td><a href="./Day_76/README.md">README</a>, <a href="./Day_76/feature_map.cu">Feature Map Code</a></td>
    </tr>
    <tr>
      <td>Day 77</td>
      <td>Advanced Gradient Computation Techniques</td>
      <td><a href="./Day_77/README.md">README</a>, <a href="./Day_77/gradient_computation.cu">Gradient Computation Code</a></td>
    </tr>
    <tr>
      <td>Day 78</td>
      <td>Multi-GPU Training Implementation</td>
      <td><a href="./Day_78/README.md">README</a>, <a href="./Day_78/multi_gpu.cu">Multi-GPU Code</a></td>
    </tr>
    <tr>
      <td>Day 79</td>
      <td>Attention Mechanism Implementation</td>
      <td><a href="./Day_79/README.md">README</a>, <a href="./Day_79/attention.cu">Attention Code</a></td>
    </tr>
    <tr>
      <td>Day 80</td>
      <td>LSTM Layer Implementation in CUDA</td>
      <td><a href="./Day_80/README.md">README</a>, <a href="./Day_80/lstm.cu">LSTM Code</a></td>
    </tr>
    <tr>
      <td>Day 81</td>
      <td>Memory Pooling for Dynamic Networks</td>
      <td><a href="./Day_81/README.md">README</a>, <a href="./Day_81/memory_pool.cu">Memory Pool Code</a></td>
    </tr>
    <tr>
      <td>Day 82</td>
      <td>Profiling and Performance Analysis</td>
      <td><a href="./Day_82/README.md">README</a>, <a href="./Day_82/profiling.cu">Profiling Code</a></td>
    </tr>
    <tr>
      <td>Day 83</td>
      <td>Custom Memory Allocator for Neural Networks</td>
      <td><a href="./Day_83/README.md">README</a>, <a href="./Day_83/allocator.cu">Allocator Code</a></td>
    </tr>
    <tr>
      <td>Day 84</td>
      <td>Quantization and Low-Precision Computing</td>
      <td><a href="./Day_84/README.md">README</a>, <a href="./Day_84/quantization.cu">Quantization Code</a></td>
    </tr>
    <tr>
      <td>Day 85</td>
      <td>Model Checkpointing and Loading</td>
      <td><a href="./Day_85/README.md">README</a>, <a href="./Day_85/checkpoint.cu">Checkpoint Code</a></td>
    </tr>
    <tr>
      <td>Day 86</td>
      <td>Advanced Loss Function Implementation</td>
      <td><a href="./Day_86/README.md">README</a>, <a href="./Day_86/loss_function.cu">Loss Function Code</a></td>
    </tr>
    <tr>
      <td>Day 87</td>
      <td>Custom Layer Integration Framework</td>
      <td><a href="./Day_87/README.md">README</a>, <a href="./Day_87/custom_layer.cu">Custom Layer Code</a></td>
    </tr>
    <tr>
      <td>Day 88</td>
      <td>Advanced Optimization Algorithms</td>
      <td><a href="./Day_88/README.md">README</a>, <a href="./Day_88/advanced_optimizer.cu">Advanced Optimizer Code</a></td>
    </tr>
    <tr>
      <td>Day 89</td>
      <td>Model Architecture Search</td>
      <td><a href="./Day_89/README.md">README</a>, <a href="./Day_89/architecture_search.cu">Architecture Search Code</a></td>
    </tr>
    <tr>
      <td>Day 90</td>
      <td>Pipeline Parallelism Implementation</td>
      <td><a href="./Day_90/README.md">README</a>, <a href="./Day_90/pipeline_parallelism.cu">Pipeline Parallelism Code</a></td>
    </tr>
    <tr>
      <td>Day 91</td>
      <td>Performance Tuning and Benchmarking</td>
      <td><a href="./Day_91/README.md">README</a>, <a href="./Day_91/benchmarking.cu">Benchmarking Code</a></td>
    </tr>
    <tr>
      <td>Day 92</td>
      <td>Memory Fragmentation Handling</td>
      <td><a href="./Day_92/README.md">README</a>, <a href="./Day_92/fragmentation.cu">Fragmentation Code</a></td>
    </tr>
    <tr>
      <td>Day 93</td>
      <td>Advanced Data Preprocessing on GPU</td>
      <td><a href="./Day_93/README.md">README</a>, <a href="./Day_93/preprocessing.cu">Preprocessing Code</a></td>
    </tr>
    <tr>
      <td>Day 94</td>
      <td>Custom CUDA Stream Management</td>
      <td><a href="./Day_94/README.md">README</a>, <a href="./Day_94/stream_management.cu">Stream Management Code</a></td>
    </tr>
    <tr>
      <td>Day 95</td>
      <td>Advanced Error Handling and Recovery</td>
      <td><a href="./Day_95/README.md">README</a>, <a href="./Day_95/error_handling.cu">Error Handling Code</a></td>
    </tr>
    <tr>
      <td>Day 96</td>
      <td>Dynamic Kernel Configuration</td>
      <td><a href="./Day_96/README.md">README</a>, <a href="./Day_96/kernel_config.cu">Kernel Config Code</a></td>
    </tr>
    <tr>
      <td>Day 97</td>
      <td>Advanced Memory Access Patterns</td>
      <td><a href="./Day_97/README.md">README</a>, <a href="./Day_97/memory_patterns.cu">Memory Patterns Code</a></td>
    </tr>
    <tr>
      <td>Day 98</td>
      <td>Performance Optimization Case Studies</td>
      <td><a href="./Day_98/README.md">README</a>, <a href="./Day_98/case_study.cu">Case Study Code</a></td>
    </tr>
    <tr>
      <td>Day 99</td>
      <td>Final Project: Complete DL Framework</td>
      <td><a href="./Day_99/README.md">README</a>, <a href="./Day_99/framework.cu">Framework Code</a></td>
    </tr>
    <tr>
      <td>Day 100</td>
      <td>Advanced 2D Operations and Optimizations</td>
      <td><a href="./Day_100/README.md">README</a>, <a href="./Day_100/2d_max.cu">2D Max Code</a></td>
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

