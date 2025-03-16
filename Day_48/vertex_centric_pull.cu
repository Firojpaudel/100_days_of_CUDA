#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <climits>
using namespace std;

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            cerr << "CUDA Error: " << cudaGetErrorString(err)      \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

struct CSRGraph
{
    unsigned int numVertices;
    unsigned int *srcPtrs;
    unsigned int *dst;
};

__global__ void bfs_bottom_up_kernel(CSRGraph csrGraph, unsigned int *level,
                                     unsigned int *newVertexVisited, unsigned int currLevel)
{
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < csrGraph.numVertices)
    {
        if (level[vertex] == UINT_MAX)
        {
            for (unsigned int edge = csrGraph.srcPtrs[vertex];
                 edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
            {
                unsigned int neighbor = csrGraph.dst[edge];
                if (level[neighbor] == currLevel - 1)
                {
                    level[vertex] = currLevel;
                    atomicExch(newVertexVisited, 1U);
                    break;
                }
            }
        }
    }
}

int main()
{
    unsigned int numVertices = 6;
    vector<unsigned int> srcPtrs_host = {0, 2, 3, 4, 4, 4, 4};
    vector<unsigned int> dst_host = {1, 2, 3, 4};
    // Transpose graph for bottom-up BFS
    vector<unsigned int> srcPtrs_trans_host = {0, 0, 1, 2, 3, 4, 4};
    vector<unsigned int> dst_trans_host = {0, 0, 1, 2};

    cout << "Starting BFS on graph with " << numVertices << " vertices." << endl;

    // Device memory allocation
    unsigned int *d_srcPtrs, *d_dst, *d_level, *d_newVertexVisited;
    unsigned int *d_srcPtrs_trans, *d_dst_trans;
    CUDA_CHECK(cudaMalloc(&d_srcPtrs, (numVertices + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_dst, dst_host.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_srcPtrs_trans, (numVertices + 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_dst_trans, dst_trans_host.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_level, numVertices * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_newVertexVisited, sizeof(unsigned int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_srcPtrs, srcPtrs_host.data(),
                          (numVertices + 1) * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst, dst_host.data(),
                          dst_host.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_srcPtrs_trans, srcPtrs_trans_host.data(),
                          (numVertices + 1) * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst_trans, dst_trans_host.data(),
                          dst_trans_host.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Initialize levels
    CUDA_CHECK(cudaMemset(d_level, 0xFF, numVertices * sizeof(unsigned int)));
    unsigned int zero = 0;
    unsigned int start = 0;
    CUDA_CHECK(cudaMemcpy(&d_level[start], &zero, sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    // Configure transpose CSRGraph
    CSRGraph transGraph;
    transGraph.numVertices = numVertices;
    transGraph.srcPtrs = d_srcPtrs_trans;
    transGraph.dst = d_dst_trans;

    // BFS loop
    unsigned int currLevel = 1;
    bool continueBFS = true;
    while (continueBFS)
    {
        cout << "Processing level " << currLevel << "..." << endl;
        CUDA_CHECK(cudaMemset(d_newVertexVisited, 0, sizeof(unsigned int)));

        int blockSize = 256;
        int gridSize = (numVertices + blockSize - 1) / blockSize;
        bfs_bottom_up_kernel<<<gridSize, blockSize>>>(transGraph, d_level,
                                                      d_newVertexVisited, currLevel);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int newVisited;
        CUDA_CHECK(cudaMemcpy(&newVisited, d_newVertexVisited,
                              sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (newVisited == 0)
        {
            cout << "No new vertices visited at level " << currLevel
                 << ", BFS complete." << endl;
            continueBFS = false;
        }
        else
        {
            cout << "New vertices visited, moving to level " << currLevel + 1
                 << "." << endl;
            currLevel++;
        }
    }

    // Copy results to host
    vector<unsigned int> level_host(numVertices);
    CUDA_CHECK(cudaMemcpy(level_host.data(), d_level,
                          numVertices * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // Print results
    cout << "\nBFS Results:" << endl;
    for (unsigned int i = 0; i < numVertices; ++i)
    {
        cout << "Vertex " << i << ": Level ";
        if (level_host[i] == UINT_MAX)
        {
            cout << "Unreachable" << endl;
        }
        else
        {
            cout << level_host[i] << endl;
        }
    }

    // Free memory
    CUDA_CHECK(cudaFree(d_srcPtrs));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_srcPtrs_trans));
    CUDA_CHECK(cudaFree(d_dst_trans));
    CUDA_CHECK(cudaFree(d_level));
    CUDA_CHECK(cudaFree(d_newVertexVisited));

    return 0;
}