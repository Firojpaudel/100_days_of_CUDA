# 100_days_of_CUDA
Challenging myself to learn CUDA (Basics ⇾ Intermediate) these 100 days. 

My learning resources: 
1. **Books**:
    - **Cuda By Example** _An Introduction to General-Purpose GPU Programming_ — Jason Sandres, Edward Kandrot
    - **PMPP**; _*4th Edition_ — Wen-mei, David, Izzat
#### Progress: 

<div align="center">

| Days  | Learnt Topics | Link/s |
|-------|---------------|------|
| Day 01 | History, Applications, Setup, First Hello World CUDA program | [README](./Day_01/README.md), [Code](./Day_01/hello.cu) |
| Day 02 | Parameters Passing, Device Queries, Vector Addition on Kernel, PMPP Chapter_02 Exercises Solved| [README](./Day_02/README.md), [Params_passing Code](./Day_02/params.cu), [Dev_Query Code](./Day_02/dev_queries.cu), [Vect_addn Code](./Day_02/vect_addn.cu)|
| Day 03 | Multidimensional Grids Organization, Mapping Threads to Multidimensional Data | [README](./Day_03/README.md), [Code for Grids Explanation](./Day_03/grids.cu), [RGB to Grayscale Conversion Code](./Day_03/image_color_conv.cu) |
| Day 04 | Image Blurring, Matrix Multiplication, Solution to Exercise 1(a,b) and 2 | [README](./Day_04/README.md), [Image Blurring Code](./Day_04/image_blur.cu), [Exercise_Qn.1(a) Code](./Day_04/Exercise_01_soln_a.cu), [Exercise_Qn.1(b) Code](./Day_04/Exercise_01_soln_b.cu), [Exercise 2 Code](./Day_04/Exercise_02_soln.cu)|
| Day 05 | Architecture of modern GPU _(intro)_, Architecture diagram understanding, Block Scheduling, Barrier Synchronization and using `__syncthreads()`| [README](./Day_05/README.md), [Barrier Synchronization Code](./Day_05/barrier_sync.cu) |
| Day 06 | Warps and SIMD Hardware, The modern GPU architecture, Control Divergence _Intro_ | [README](./Day_06/README.md) | 
| Day 07 | Studying impacts of divergence on Performance, Types of divergence _(If-else & Loop-Based)_, Divergence identification, Performance Impace Analysis | [README](./Day_07/README.md), [if_else divergence Code](./Day_07/if-else_diverge.cu), [loops_warp_divergence Code](./Day_07/loops_warp_divergence.cu) |
| Day 08 | Warp Scheduling and Latency Tolerance, Resource Partitioning & Occupancy | [README](./Day_08/README.md), [Exercise_02_Solution Code](./Day_08/Exercise_02.cu), [Exercise_04_Solution Code](./Day_08/Exercise_04.cu) |
| Day 09 | Memory Access Effeciency in CUDA, Roofline Model, Matrix Multiplication Code Optimization | [README](./Day_09/README.md), [Matrix Multiplication Code](./Day_09/matrix_multiplication.cu), [Optimized Matrix Multiplication Code](./Day_09/optimized_mat_mul.cu) |
| Day 10 | CUDA Memory Types: Global, Constant, Local, Registers, Shared | [README](./Day_10/README.md), [Memory Types Code](./Day_10/mem_types_in_action.cu) |
| Day 11 | Tiling Concept and Memory Tradeoffs | [README](./Day_11/README.md), [Tiled Matrix Multiplication Code](./Day_11/tiled_mat_mul.cu) |
| Day 12 | Explanation for Day 11 Tiled Matrix Multiplication Code, Impact of Memory Usage on Occupancy | [README](./Day_12/README.md), [Dynamic Tiled Matrix Multiplication Kernel Code](./Day_12/Day_12_updated_code.cu) |
| Day 13 | Memory Coalescing in CUDA, Row-Major vs. Column-Major Storage, Coalsced Memory Access in CUDA, Understanding DRAM and Burst Access | [README](./Day_13/README.md), [ Row VS Column Majors Code](./Day_13/row_vs_column_major.cu)|
| Day 14 | Explaining the Corner Turning in Mat Mul, Memory Coalescing with a bit of Analogy, Memory Latency Hiding | [ README](./Day_14/README.md), [Code for Corner Turning](./Day_14/corner_turning.cu)|
| Day 15 | Thread Coarsening, Exercises of Chapter 6 of PMPP| [README](./Day_15/README.md), [Code For Thread Coarsening](./Day_15/thread_coarsening.cu) |
| Day 16 | Start of Chapter 7: Convolutions; 1D and 2D Convolution with Boundary Conditions | [README](./Day_16/README.md), [ Code For 1D Convolution](./Day_16/1D_Conv.cu),[Code For 2D Convolution](./Day_16/2D_Conv.cu)|
| Day 17 | Parallel 2D Convolution implementation with Edge Handling, Normalization | [README](./Day_17/README.md), [Code For 2D Convolution with Edge Handlings](./Day_17/2D_convo.cu)|
| Day 18 | Implementation of Convolution on 2D image | [README](./Day_18/README.md), [Image Preprocessing Code](./Day_18/Convolution/prepare.py), [CUDA Convolution Kernel Code](./Day_18/Convolution/Convolution_img.cu), [Code For Post_Processing and Displaying](./Day_18/Convolution/post_processing.py)|
| Day 19 | Properties of Filter Array in Conv, Constant memory in CUDA, Caching in CUDA and Memory Hierarchy, Tiled Convolution with Halo Cells, Thread Organization Strategies | [README](./Day_19/README.md), [Tiled 2D convolution Code](./Day_19/tiled_2D_conv.cu) |
| Day 20 |  Tiled Convolution Using Caches for Halo Cells , Exercises from Chapter 7 | [README](./Day_20/README.md), [Tiled 2D Conv Code](./Day_20/tiled_2D_with_cache.cu), [3D Convolution basic kernel](./Day_20/3D.cu) |
| Day 21 | Chapter 8: **Stencil**, Differenece between Stencil and Convolution, Parallel Stencil (Algos) | [README](./Day_21/README.md), [Basic Stencil Code](./Day_21/basic_stencil.cu), [Optimized Stencil Code](./Day_21/optimized_stencil.cu) |
| Day 22 | Thread Coarsening and optimizing 3D Stencil computations through it, Thread Coarsening Architecture | [README](./Day_22/README.md) , [3D Stencil: Unoptimized Code](./Day_22/thread_coarsening1.cu), [3D Stencil: Optimized Code](./Day_22/thread_coarsening2.cu) |
| Day 23 | Exercises from Chapter 8; Chapter Completion | [README](./Day_23/README.md)|
| Day 24 | Starting Chapter 9: **Parallel Histogram** Introudction with code| [Parallel Historam Code](./Day_24/parallel_hist.cu) |
| Day 25 | Atomic Ops | [README](./Day_25/README.md), [Privatization Code](./Day_25/privatization.cu), [Coarsening Code](./Day_25/coarserning.cu), [Aggregation Code](./Day_25/aggregation.cu) |
| Day 26 | Ending of Chapter 9: Exercises, Chapter 10: **Reduction** Start | [README](./Day_26/README.md), [Max Reduction Code](./Day_26/max_reduction.cu), [Sum Reduction Code](./Day_26/sum_reduction.cu) |
| Day 27 | Simple Sum Reduction Kernel, Convergent Sum Reduction | [README](./Day_27/README.md), [Simple Sum Reduction Code](./day_27/SimpleSumReductionKernel.cu), [Convergent Sum Reduction Code](./day_27/optimizedKernel.cu) |
| Day 28 | Shared Memory For Reduction, Hierarchial Reduction, Thread Coarsening for Reduced Overheads | [README](./Day_28/README.md), [Shared Memory Reduction Code](./Day_28/shared_mem.cu), [Hierarchial Reduction Code](./Day_28/hierarchial.cu), [Coarsening Reduction Code](./Day_28/Coarsened.cu) | 
| Day 29 | Exercises from Chapter 10 | [README](./Day_29/README.md) |
| Day 30 | Parallel Prefix Scan, Kogge-Stone Parallel Prefix Scan Algo | [README](./Day_30/README.md), [Kogge Stone Code](./Day_30/koggle_stone.cu) |
| Day 31 | Kogge Stone Continue, Complexity Analysis (Both Exclusive and Inclusive) | [README](./Day_31/README.md), [Exclusive Scan Code](./Day_31/exclusive_scan.cu), [Inclusive Scan Code](./Day_31/inclusive_scan.cu) |
| Day 32 | Brent- Kung Parallel Inclusive Scan Algo | [README](./Day_32/README.md), [Brent Kung Code](./Day_32/Brent_kung.cu) |
| Day 33 | Coarsening in Detail | [README](./Day_33/README.md) |
| Day 34 | Coarsening Complexity Analysis, Hierarchial Scan | [README](./Day_34/README.md), [Coarsening Code](./Day_34/coarsening.cu), [Hierarchial Scan Code](./Day_34/hierarchial.cu) |
| Day 35 | Exercises from Chapter 11 | [README](./Day_35/README.md) |
| Day 36 | Chapter 12: Merge, Sequential Merge | [README](./Day_36/README.md), [Sequential Merge Code](./Day_36/seq_merge.cu) |
| Day 37 | Parallel Merge Kernels and Co-ranks | [README](./Day_37/README.md), [CoRank Implementation Code](./Day_37/corank.cu), [Divide and Conquer Code](./Day_37/div_conq.cu) |
| Day 38 | Tiled Merge Kernels | [README](./Day_38/README.md), [Tiled Merge Code](./Day_38/tiled_merged.cu) |
| Day 39 | Exercises from Chapter 12 | [README](./Day_39/README.md) |
| Day 40 | Paralel Radix Sort | [README](./Day_40/README.md), [Radix Sort Code](./Day_40/parallel_radix.cu) |
| Day 41 | Choice of Radix, Multi-bit radix, Optimizing Memory Coalescening Using Parallel Radix Sort, Thread Coarsening To Improve Memory Coalescening, Parallel Merge Sort| [README](./Day_41/README.md), [Coarsening Applied Code](./Day_41/coarsening_applied.cu), [Parallel Merge](./Day_41/parallel_merge_sort.cu) | 
| Day 42 | Exercises from Chapter 13 | [README](./Day_42/README.md) |
| Day 43 | SmPV with COO | [README](./Day_43/README.md), [SmPV COO Code Implementation](./Day_43/SpMV_COO.cu) |
| Day 44 | CSR Format, ELL Format | [README](./Day_44/README.md), [CSR Implementation Code](./Day_44/CSR_Implementation.cu) |
| Day 45 | Hybrid ELL-COO Format, JDS Format with parallelization | [README](./Day_45/README.md), [Hybrid ELL-COO Code](./Day_45/ELL_COO.cu) |
| Day 46 | Exercises from Chapter 14 | [README](./Day_46/README.md) |
| Day 47 | Chapter 15 Starts, Normal BFS | [README](./Day_47/README.md), [Normal BFS Code](./Day_47/bfs_simple.cu) |
| Day 48 | Vertex Centric Parallelization, Vertex Centric Pull; Push | [README](./Day_48/README.md), [Vertex Centric Code](./Day_48/vertex_centric_pull.cu), [Vertex Centric Push Code](./Day_48/vertex_centric_push.cu) |
| Day 49 | Edge Centric Pararallelization, Frontiers | [README](./Day_49/README.md), [Edge Centric Code](./Day_49/edge_centric.cu), [Frontier Code](./Day_49/frontier.cu) |
| Day 50 | Privatization, Exercises from Chapter 15 | [README](./Day_50/README.md) , [Privatization Code](./Day_50/privatization.cu)|
| Day 51 | --- Will Update in a while --- | --- |
</div>
