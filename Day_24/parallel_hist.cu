#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

#define INTERVAL_SIZE 4
#define NUM_INTERVALS 7 // Since 'a' to 'z' can be divided into 7 groups of 4 letters
#define THREADS_PER_BLOCK 256

__global__ void compute_histogram(char *data, int *histo, int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < length) {
        char c = data[idx];
        if (c >= 'a' && c <= 'z') {
            int alphabet_position = c - 'a';
            int interval_index = alphabet_position / INTERVAL_SIZE;
            atomicAdd(&histo[interval_index], 1);
        }
    }
}

int main() {
    char h_data[1024];
    printf("Enter the text (Limited to 100 Characters): ");
    fgets(h_data, 1024, stdin);
    int length = strlen(h_data);
    if (h_data[length - 1] == '\n') {
        h_data[length - 1] = '\0';
        length--;
    }

    // Allocate memory on device
    char *d_data;
    int *d_histo;
    int h_histo[NUM_INTERVALS] = {0};

    cudaMalloc((void **)&d_data, length * sizeof(char));
    cudaMalloc((void **)&d_histo, NUM_INTERVALS * sizeof(int));

    cudaMemcpy(d_data, h_data, length * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemset(d_histo, 0, NUM_INTERVALS * sizeof(int));

    // Launch Kernel
    int numBlocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_histogram<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, d_histo, length);

    // Copy result back to host
    cudaMemcpy(h_histo, d_histo, NUM_INTERVALS * sizeof(int), cudaMemcpyDeviceToHost);

    // Print histogram
    printf("\nHistogram Data:\n");
    for (int i = 0; i < NUM_INTERVALS; i++) {
        char start_char = 'a' + i * INTERVAL_SIZE;
        char end_char = start_char + INTERVAL_SIZE - 1;
        printf("Interval %d-%d [%c-%c]: %d occurrences\n",
               i * INTERVAL_SIZE, (i + 1) * INTERVAL_SIZE - 1, start_char, end_char, h_histo[i]);
    }

    // Save as CSV (Formatted Properly)
    FILE *file = fopen("histogram.csv", "w");
    if (file) {
        fprintf(file, "Interval,Start_Char,End_Char,Occurrences\n");
        for (int i = 0; i < NUM_INTERVALS; i++) {
            char start_char = 'a' + i * INTERVAL_SIZE;
            char end_char = start_char + INTERVAL_SIZE - 1;
            fprintf(file, "%d-%d,%c,%c,%d\n", i * INTERVAL_SIZE, 
                    (i + 1) * INTERVAL_SIZE - 1, start_char, end_char, h_histo[i]);
        }
        fclose(file);
        printf("\nHistogram saved to 'histogram.csv'.\n");
    } else {
        printf("Failed to open file for writing.\n");
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_histo);

    return 0;
}