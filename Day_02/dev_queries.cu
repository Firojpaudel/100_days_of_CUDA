#include <iostream>
using namespace std;

int main(void)
{
    cudaDeviceProp prop;

    int count;

    cudaGetDeviceCount(&count);

    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("------General info for device %d------\n", i);
        printf("Device Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock Rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if (prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel Exectuion Timeout: ");
        if (prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("------Memory info for device %d------\n", i);
        printf("Total global mem:  %zu\n", prop.totalGlobalMem);
        printf("Total constant Mem:  %zu\n", prop.totalConstMem);
        printf("Max mem pitch:  %zu\n", prop.memPitch);
        printf("Texture Alignment:  %zu\n", prop.textureAlignment);
        printf("   ------ MP Information for device %d ------\n", i);
        printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp:  %zu\n", prop.sharedMemPerBlock);
        printf("Registers per mp:  %d\n", prop.regsPerBlock);
        printf("Threads in warp:  %d\n", prop.warpSize);
        printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }
}

/*
Output I got:

------General info for device 0------
Device Name: NVIDIA GeForce RTX 3050 Ti Laptop GPU
Compute Capability: 8.6
Clock Rate: 1695000
Device copy overlap: Enabled
Kernel Exectuion Timeout: Enabled

------Memory info for device 0------
Total global mem:  4294443008
Total constant Mem:  65536
Max mem pitch:  2147483647
Texture Alignment:  512

------ MP Information for device 0 ------
Multiprocessor count:  20
Shared mem per mp:  49152
Registers per mp:  65536
Threads in warp:  32
Max threads per block:  1024
Max thread dimensions:  (1024, 1024, 64)
Max grid dimensions:  (2147483647, 65535, 65535)

*/