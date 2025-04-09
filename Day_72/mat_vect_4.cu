__global__ void mat_vec_multiply(const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output_c, size_t M, size_t K) {
    extern __shared__ float partial_sums[];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int warp_id = tx / 32;
    int lane_id = tx % 32;
    int number_of_warps = blockDim.x / 32;

    if (row < M) {
        float sum = 0.0f;
        for (int k = tx; k < K; k += blockDim.x) {
            sum += input_a[row * K + k] * input_b[k];
        }
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) {
            partial_sums[warp_id] = sum;
        }
        __syncthreads();
        if (tx == 0) {
            float total_sum = 0.0f;
            for (int i = 0; i < number_of_warps; i++) {
                total_sum += partial_sums[i];
            }
            output_c[row] = total_sum;
        }
    }
}

extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t M, size_t K) {
    const int BLOCKSIZE_X = 512;
    const int BLOCKSIZE_Y = 1;
    dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
    dim3 blocksPerGrid(1, M);
    size_t shared_mem_size = (BLOCKSIZE_X / 32) * sizeof(float);  // e.g., 16 * 4 = 64 bytes for 512
    mat_vec_multiply<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(input_a, input_b, output_c, M, K);
    cudaDeviceSynchronize();
}