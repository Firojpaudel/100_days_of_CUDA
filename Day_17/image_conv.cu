#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// CUDA kernel for 2D convolution
__global__ void conv2D(float *input, float *output, float *filter, int width, int height, int filter_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int r = filter_size / 2;
    float sum = 0.0;

    if (x >= r && x < width - r && y >= r && y < height - r) {
        for (int fy = -r; fy <= r; fy++) {
            for (int fx = -r; fx <= r; fx++) {
                int img_x = x + fx;
                int img_y = y + fy;
                sum += input[img_y * width + img_x] * filter[(fy + r) * filter_size + (fx + r)];
            }
        }
        output[y * width + x] = sum;
    }
}

// Host function to run the CUDA kernel
void applyConvolutionCUDA(const Mat &input, Mat &output, float *h_filter, int filter_size) {
    int width = input.cols;
    int height = input.rows;

    // Allocate memory on GPU
    float *d_input, *d_output, *d_filter;
    cudaMalloc((void **)&d_input, width * height * sizeof(float));
    cudaMalloc((void **)&d_output, width * height * sizeof(float));
    cudaMalloc((void **)&d_filter, filter_size * filter_size * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_input, input.ptr<float>(), width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, filter_size * filter_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    conv2D<<<gridSize, blockSize>>>(d_input, d_output, d_filter, width, height, filter_size);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(output.ptr<float>(), d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}

// Main function
int main() {
    // Load grayscale image
    Mat image = imread("./images/Charmander.png", IMREAD_GRAYSCALE);
    if (image.empty()) {
        printf("Error: Image not found!\n");
        return -1;
    }
    image.convertTo(image, CV_32F, 1.0 / 255); // Normalize

    // Define filter (3x3 Box Blur example)
    float h_filter[] = {
        1/9.0, 1/9.0, 1/9.0,
        1/9.0, 1/9.0, 1/9.0,
        1/9.0, 1/9.0, 1/9.0
    };

    // Prepare output matrix
    Mat output(image.rows, image.cols, CV_32F, Scalar(0));

    // Apply convolution
    applyConvolutionCUDA(image, output, h_filter, 3);

    // Display results
    imshow("Input Image", image);
    imshow("Convolved Image", output);
    waitKey(0);

    return 0;
}