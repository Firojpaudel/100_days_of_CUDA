__global__ void diagonalMatMulKernel(const float* diagonal_a, const float* input_b, 
    float* output_c, size_t n, size_t m) {
__shared__ float shared_diag[5]; // Matches blockDim.y

size_t row = blockIdx.y * blockDim.y + threadIdx.y;
size_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 8; // Process 8 columns

// Load diagonal_a into shared memory
if (threadIdx.x == 0 && row < n) {
shared_diag[threadIdx.y] = diagonal_a[row];
}
__syncthreads();

if (row < n && col < m) {
float diag_val = shared_diag[threadIdx.y];
size_t idx = row * m + col;

// Vectorized load/store for 8 floats
if (col + 7 < m) {
float4 b_vals1 = *reinterpret_cast<const float4*>(&input_b[idx]);
float4 b_vals2 = *reinterpret_cast<const float4*>(&input_b[idx + 4]);
float4 c_vals1, c_vals2;
c_vals1.x = diag_val * b_vals1.x;
c_vals1.y = diag_val * b_vals1.y;
c_vals1.z = diag_val * b_vals1.z;
c_vals1.w = diag_val * b_vals1.w;
c_vals2.x = diag_val * b_vals2.x;
c_vals2.y = diag_val * b_vals2.y;
c_vals2.z = diag_val * b_vals2.z;
c_vals2.w = diag_val * b_vals2.w;
*reinterpret_cast<float4*>(&output_c[idx]) = c_vals1;
*reinterpret_cast<float4*>(&output_c[idx + 4]) = c_vals2;
} else {
#pragma unroll
for (size_t i = 0; i < 8 && (col + i) < m; ++i) {
output_c[idx + i] = diag_val * input_b[idx + i];
}
}
}
}

extern "C" void solution(const float* diagonal_a, const float* input_b, 
float* output_c, size_t n, size_t m) {
dim3 blockDim(32, 5);
dim3 gridDim(((m + 7) / 8 + blockDim.x - 1) / blockDim.x, 
(n + blockDim.y - 1) / blockDim.y);
diagonalMatMulKernel<<<gridDim, blockDim>>>(diagonal_a, input_b, output_c, n, m);
}