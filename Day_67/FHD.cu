#include <cuda_runtime.h>
#include <stdio.h>

// Constants for thread block sizes
#define MU_THREADS_PER_BLOCK 1024
#define FHD_THREADS_PER_BLOCK 1024
#define PI 3.14159265359f

/**
 * Kernel 1: Compute rMu and iMu (intermediate complex vector Mu = Phi * D)
 * 
 * This kernel parallelizes the computation of rMu and iMu for each k-space sample.
 * Each thread handles one k-space sample, computing its contribution independently.
 * 
 * Parameters:
 *   rPhi  - Real part of the phase term Phi (size M)
 *   iPhi  - Imaginary part of the phase term Phi (size M)
 *   rD    - Real part of the k-space data D (size M)
 *   iD    - Imaginary part of the k-space data D (size M)
 *   rMu   - Real part of the intermediate vector Mu (size M, output)
 *   iMu   - Imaginary part of the intermediate vector Mu (size M, output)
 *   M     - Number of k-space samples
 */
__global__ void cmpMu(float* rPhi, float* iPhi, float* rD, float* iD, float* rMu, float* iMu, int M) {
    // Compute global thread index (one thread per k-space sample)
    int m = blockIdx.x * MU_THREADS_PER_BLOCK + threadIdx.x;

    // Check if the thread is within bounds
    if (m < M) {
        // Compute real and imaginary parts of Mu = Phi * D
        rMu[m] = rPhi[m] * rD[m] + iPhi[m] * iD[m];
        iMu[m] = rPhi[m] * iD[m] - iPhi[m] * rD[m];
    }
}

/**
 * Kernel 2: Compute rFHD and iFHD (F^H D) using the gather approach
 * 
 * This kernel parallelizes the computation of F^H D, where each thread computes
 * the contribution to one voxel by summing over all k-space samples. This avoids
 * atomic operations, as each thread writes to its own output element.
 * 
 * Parameters:
 *   kx    - X-coordinates of k-space samples (size M)
 *   ky    - Y-coordinates of k-space samples (size M)
 *   kz    - Z-coordinates of k-space samples (size M)
 *   x     - X-coordinates of voxels (size N)
 *   y     - Y-coordinates of voxels (size N)
 *   z     - Z-coordinates of voxels (size N)
 *   rMu   - Real part of the intermediate vector Mu (size M)
 *   iMu   - Imaginary part of the intermediate vector Mu (size M)
 *   rFHD  - Real part of F^H D (size N, output)
 *   iFHD  - Imaginary part of F^H D (size N, output)
 *   M     - Number of k-space samples
 *   N     - Number of voxels in the reconstructed image
 */
__global__ void cmpFHD(float* kx, float* ky, float* kz, float* x, float* y, float* z,
                       float* rMu, float* iMu, float* rFHD, float* iFHD, int M, int N) {
    // Compute global thread index (one thread per voxel)
    int n = blockIdx.x * FHD_THREADS_PER_BLOCK + threadIdx.x;

    // Check if the thread is within bounds
    if (n < N) {
        // Initialize local accumulators for the real and imaginary parts
        float rSum = 0.0f;
        float iSum = 0.0f;

        // Sum contributions from all k-space samples for this voxel
        for (int m = 0; m < M; m++) {
            // Compute the phase term for the Fourier transform
            float expFHD = 2.0f * PI * (kx[m] * x[n] + ky[m] * y[n] + kz[m] * z[n]);
            
            // Compute cosine and sine of the phase term
            float cArg = cosf(expFHD);
            float sArg = sinf(expFHD);

            // Accumulate contributions to real and imaginary parts
            rSum += rMu[m] * cArg - iMu[m] * sArg;
            iSum += iMu[m] * cArg + rMu[m] * sArg;
        }

        // Write the final result to global memory
        rFHD[n] = rSum;
        iFHD[n] = iSum;
    }
}

/**
 * Host function to compute F^H D using CUDA
 * 
 * This function manages memory allocation, data transfer, and kernel launches
 * to compute F^H D on the GPU. It uses two kernels: cmpMu to compute the
 * intermediate vector Mu, and cmpFHD to compute the final F^H D.
 * 
 * Parameters:
 *   h_rPhi  - Host pointer to real part of Phi (size M)
 *   h_iPhi  - Host pointer to imaginary part of Phi (size M)
 *   h_rD    - Host pointer to real part of D (size M)
 *   h_iD    - Host pointer to imaginary part of D (size M)
 *   h_kx    - Host pointer to k-space X-coordinates (size M)
 *   h_ky    - Host pointer to k-space Y-coordinates (size M)
 *   h_kz    - Host pointer to k-space Z-coordinates (size M)
 *   h_x     - Host pointer to voxel X-coordinates (size N)
 *   h_y     - Host pointer to voxel Y-coordinates (size N)
 *   h_z     - Host pointer to voxel Z-coordinates (size N)
 *   h_rFHD  - Host pointer to real part of F^H D (size N, output)
 *   h_iFHD  - Host pointer to imaginary part of F^H D (size N, output)
 *   M       - Number of k-space samples
 *   N       - Number of voxels in the reconstructed image
 */
void computeFHD(float* h_rPhi, float* h_iPhi, float* h_rD, float* h_iD,
                float* h_kx, float* h_ky, float* h_kz,
                float* h_x, float* h_y, float* h_z,
                float* h_rFHD, float* h_iFHD, int M, int N) {
    // Device pointers for all arrays
    float *d_rPhi, *d_iPhi, *d_rD, *d_iD;
    float *d_kx, *d_ky, *d_kz;
    float *d_x, *d_y, *d_z;
    float *d_rMu, *d_iMu;
    float *d_rFHD, *d_iFHD;

    // Allocate device memory for all arrays
    cudaMalloc(&d_rPhi, M * sizeof(float));
    cudaMalloc(&d_iPhi, M * sizeof(float));
    cudaMalloc(&d_rD, M * sizeof(float));
    cudaMalloc(&d_iD, M * sizeof(float));
    cudaMalloc(&d_kx, M * sizeof(float));
    cudaMalloc(&d_ky, M * sizeof(float));
    cudaMalloc(&d_kz, M * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_rMu, M * sizeof(float));
    cudaMalloc(&d_iMu, M * sizeof(float));
    cudaMalloc(&d_rFHD, N * sizeof(float));
    cudaMalloc(&d_iFHD, N * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_rPhi, h_rPhi, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iPhi, h_iPhi, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rD, h_rD, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iD, h_iD, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kx, h_kx, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ky, h_ky, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kz, h_kz, M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch first kernel: compute rMu, iMu
    int muBlocks = (M + MU_THREADS_PER_BLOCK - 1) / MU_THREADS_PER_BLOCK;
    cmpMu<<<muBlocks, MU_THREADS_PER_BLOCK>>>(d_rPhi, d_iPhi, d_rD, d_iD, d_rMu, d_iMu, M);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cmpMu kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Launch second kernel: compute rFHD, iFHD
    int fhdBlocks = (N + FHD_THREADS_PER_BLOCK - 1) / FHD_THREADS_PER_BLOCK;
    cmpFHD<<<fhdBlocks, FHD_THREADS_PER_BLOCK>>>(d_kx, d_ky, d_kz, d_x, d_y, d_z,
                                                 d_rMu, d_iMu, d_rFHD, d_iFHD, M, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cmpFHD kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Synchronize to ensure kernels complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_rFHD, d_rFHD, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iFHD, d_iFHD, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rPhi); cudaFree(d_iPhi); cudaFree(d_rD); cudaFree(d_iD);
    cudaFree(d_kx); cudaFree(d_ky); cudaFree(d_kz);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_rMu); cudaFree(d_iMu);
    cudaFree(d_rFHD); cudaFree(d_iFHD);
}

/**
 * Example main function to demonstrate usage
 * 
 * This main function allocates dummy data, calls computeFHD, and prints a small
 * portion of the results. In a real application, the input arrays would be filled
 * with actual k-space and voxel data.
 */
int main() {
    // Example sizes
    int M = 1000;    // Number of k-space samples
    int N = 128 * 128 * 128;  // Number of voxels (128^3)

    // Allocate host memory
    float *h_rPhi = (float*)malloc(M * sizeof(float));
    float *h_iPhi = (float*)malloc(M * sizeof(float));
    float *h_rD = (float*)malloc(M * sizeof(float));
    float *h_iD = (float*)malloc(M * sizeof(float));
    float *h_kx = (float*)malloc(M * sizeof(float));
    float *h_ky = (float*)malloc(M * sizeof(float));
    float *h_kz = (float*)malloc(M * sizeof(float));
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    float *h_z = (float*)malloc(N * sizeof(float));
    float *h_rFHD = (float*)malloc(N * sizeof(float));
    float *h_iFHD = (float*)malloc(N * sizeof(float));

    // Initialize dummy data (in a real application, fill with actual data)
    for (int m = 0; m < M; m++) {
        h_rPhi[m] = 1.0f;
        h_iPhi[m] = 0.0f;
        h_rD[m] = 1.0f;
        h_iD[m] = 0.0f;
        h_kx[m] = m * 0.001f;
        h_ky[m] = m * 0.001f;
        h_kz[m] = m * 0.001f;
    }
    for (int n = 0; n < N; n++) {
        h_x[n] = n * 0.001f;
        h_y[n] = n * 0.001f;
        h_z[n] = n * 0.001f;
        h_rFHD[n] = 0.0f;
        h_iFHD[n] = 0.0f;
    }

    // Call the computeFHD function
    computeFHD(h_rPhi, h_iPhi, h_rD, h_iD, h_kx, h_ky, h_kz,
               h_x, h_y, h_z, h_rFHD, h_iFHD, M, N);

    // Print a small portion of the results
    printf("First 5 elements of rFHD:\n");
    for (int n = 0; n < 5; n++) {
        printf("rFHD[%d] = %f\n", n, h_rFHD[n]);
    }
    printf("First 5 elements of iFHD:\n");
    for (int n = 0; n < 5; n++) {
        printf("iFHD[%d] = %f\n", n, h_iFHD[n]);
    }

    // Free host memory
    free(h_rPhi); free(h_iPhi); free(h_rD); free(h_iD);
    free(h_kx); free(h_ky); free(h_kz);
    free(h_x); free(h_y); free(h_z);
    free(h_rFHD); free(h_iFHD);

    return 0;
}