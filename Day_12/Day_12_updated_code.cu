#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void dynamicTiledMulKernel(float* M, float* N, float* P, int Width, int tile_width, size_t Mds_size) {
    extern __shared__ char shared_mem[];
    float* Mds = (float*)shared_mem;
    float* Nds = (float*)&shared_mem[Mds_size];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int Row = by * tile_width + ty;
    int Col = bx * tile_width + tx;
    
    float Pvalue = 0.0f;

    for (int ph = 0; ph < (Width + tile_width - 1)/tile_width; ++ph) {
        // Load tiles into shared memory
        int M_idx = Row * Width + ph * tile_width + tx;
        int N_idx = (ph * tile_width + ty) * Width + Col;
        
        if (Row < Width && (ph * tile_width + tx) < Width)
            Mds[ty * tile_width + tx] = M[M_idx];
        else
            Mds[ty * tile_width + tx] = 0.0f;

        if (Col < Width && (ph * tile_width + ty) < Width)
            Nds[ty * tile_width + tx] = N[N_idx];
        else
            Nds[ty * tile_width + tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < tile_width; ++k) {
            Pvalue += Mds[ty * tile_width + k] * Nds[k * tile_width + tx];
        }
        __syncthreads();
    }

    if (Row < Width && Col < Width)
        P[Row * Width + Col] = Pvalue;
}

void cpuMatMul(float *A, float *B, float *C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i*N + j] += A[i*N + k] * B[k*N + j];
}

int main() {
    std::cout << "_________________________________________\n"
              << "  Adaptive Matrix Multiplication Analyzer \n"
              << "__________________________________________\n";

    int N;
    std::cout << "Enter matrix dimension: ";
    std::cin >> N;

    if(N <= 0) {
        std::cerr << "Error: Matrix size must be positive!\n";
        return EXIT_FAILURE;
    }

    // Get GPU properties
    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0), "Get device properties");
    
    // Calculate optimal tile size
    size_t max_tile_elements = prop.sharedMemPerBlock / (2 * sizeof(float));
    int tile_width = static_cast<int>(sqrt(max_tile_elements));
    
    // Consider max threads per block
    tile_width = std::min(tile_width, (int)sqrt(prop.maxThreadsPerBlock));
    tile_width = std::min(tile_width, 32); // Common optimal size
    
    size_t shared_mem_size = 2 * tile_width * tile_width * sizeof(float);
    
    std::cout << "\nGPU Configuration:\n";
    std::cout << " - Max Shared Memory/Block: " << prop.sharedMemPerBlock << " bytes\n";
    std::cout << " - Selected Tile Size: " << tile_width << "x" << tile_width << "\n";
    std::cout << " - Shared Memory Usage: " << shared_mem_size << " bytes/block\n";

    // Memory allocation
    size_t bytes = N * N * sizeof(float);
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *cpuRes = new float[N*N]{};
    float *gpuRes = new float[N*N]{};

    // Initialize matrices
    std::cout << "\nInitializing matrices...\n";
    for(int i = 0; i < N*N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Benchmarking
    double cpuTime = 0, gpuTime = 0;
    bool useGPU = N > 256; // Dynamic threshold based on device properties

    // CPU Execution
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuMatMul(A, B, cpuRes, N);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration<double>(cpuEnd - cpuStart).count();

    if(useGPU) {
        // GPU Memory management
        float *dM, *dN, *dP;
        checkCudaError(cudaMalloc(&dM, bytes), "Allocate dM");
        checkCudaError(cudaMalloc(&dN, bytes), "Allocate dN");
        checkCudaError(cudaMalloc(&dP, bytes), "Allocate dP");

        checkCudaError(cudaMemcpy(dM, A, bytes, cudaMemcpyHostToDevice), "Copy H2D M");
        checkCudaError(cudaMemcpy(dN, B, bytes, cudaMemcpyHostToDevice), "Copy H2D N");

        // Kernel configuration
        dim3 blocks((N + tile_width - 1)/tile_width, (N + tile_width - 1)/tile_width);
        dim3 threads(tile_width, tile_width);
        
        cudaEvent_t start, stop;
        checkCudaError(cudaEventCreate(&start), "Create start event");
        checkCudaError(cudaEventCreate(&stop), "Create stop event");
        
        checkCudaError(cudaEventRecord(start), "Record start");
        
        dynamicTiledMulKernel<<<blocks, threads, shared_mem_size>>>(
            dM, dN, dP, N, tile_width, tile_width * tile_width * sizeof(float));
        
        checkCudaError(cudaGetLastError(), "Kernel launch");
        checkCudaError(cudaDeviceSynchronize(), "Device sync");
        
        checkCudaError(cudaEventRecord(stop), "Record stop");
        checkCudaError(cudaEventSynchronize(stop), "Event sync");
        
        float ms;
        checkCudaError(cudaEventElapsedTime(&ms, start, stop), "Event elapsed");
        gpuTime = ms / 1000.0;

        checkCudaError(cudaMemcpy(gpuRes, dP, bytes, cudaMemcpyDeviceToHost), "Copy D2H");
        
        // Cleanup
        cudaFree(dM);
        cudaFree(dN);
        cudaFree(dP);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Validation
    float maxError = 0.0f;
    if(useGPU) {
        for(int i = 0; i < N*N; ++i) {
            maxError = fmax(maxError, fabs(cpuRes[i] - gpuRes[i]));
        }
    }

    // Results
    std::cout << "\n---------------- Results ---------------\n";
    std::cout << "Matrix Size:        " << N << "x" << N << "\n";
    std::cout << "Compute Device:     " << (useGPU ? "GPU" : "CPU") << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "CPU Time:           " << cpuTime << " s\n";
    if(useGPU) {
        std::cout << "GPU Time:           " << gpuTime << " s\n";
        std::cout << "Speedup Factor:     " << cpuTime/gpuTime << "x\n";
        std::cout << "Max Error:          " << std::scientific << maxError << "\n";
    }
    std::cout << "----------------------------------------\n";

    delete[] A; delete[] B; delete[] cpuRes; delete[] gpuRes;
    return EXIT_SUCCESS;
}