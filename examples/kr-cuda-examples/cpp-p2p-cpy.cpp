#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#define BUF_SIZE 1024

int main()
{
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    if (numDevices < 2)
    {
        std::cout << "At least two GPUs are required to run this example." << std::endl;
        return 0;
    }

    int *h;
    int *g0;
    int *g1;
    int *h_copy;

    h = new int[BUF_SIZE];
    for (int i = 0; i < BUF_SIZE; i++)
        h[i] = i;

    cudaSetDevice(0);
    cudaMalloc(&g0, BUF_SIZE * sizeof(int));
    cudaMemset(g0, 0, BUF_SIZE * sizeof(int));
    cudaMemcpy(g0, h, BUF_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMalloc(&g1, BUF_SIZE * sizeof(int));
    cudaMemset(g1, 0, BUF_SIZE * sizeof(int));
    cudaDeviceEnablePeerAccess(0, 0);
    cudaMemcpyPeer(g1, 1, g0, 0, BUF_SIZE * sizeof(int));

    h_copy = new int[BUF_SIZE];
    cudaSetDevice(1);
    cudaMemcpy(h_copy, g1, BUF_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < BUF_SIZE; i++)
    {
        if (h[i] != h_copy[i])
        {
            std::cout << "Mismatch at index " << i << ": expected " << h[i] << ", got " << h_copy[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success)
        std::cout << "Data copy successful." << std::endl;

    cudaSetDevice(0);
    cudaDeviceDisablePeerAccess(1);
    cudaFree(g0);
    cudaSetDevice(1);
    cudaFree(g1);

    delete[] h;
    delete[] h_copy;

    return 0;
}
