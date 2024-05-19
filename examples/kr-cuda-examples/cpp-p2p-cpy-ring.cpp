#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define BUF_SIZ 1024

__global__ void bumparray(int *array, const int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size)
        array[idx]++;
}

int main()
{
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices < 2)
    {
        std::cout << "At least two GPUs are required to run this example." << std::endl;
        return 0;
    }

    int *h = new int[BUF_SIZ];
    int *h_copy = new int[BUF_SIZ];
    for (int i = 0; i < BUF_SIZ; i++)
        h[i] = i; // initialize the buffer to conduct the test

    // for the gpus
    int **g;
    g = new int *[numDevices]; // get an array of pointers to integers
    // for (int i = 0; i < numDevices; i++) g[i] = new int[BUF_SIZ];

    // now start by having host copy array to first gpu
    cudaSetDevice(0);
    cudaMalloc(&g[0], BUF_SIZ * sizeof(int));
    cudaMemset(&g[0][0], 0, BUF_SIZ * sizeof(int));
    cudaMemcpy(&g[0][0], h, BUF_SIZ * sizeof(int), cudaMemcpyHostToDevice);

    // call the update function from gpu 0
    //... it will be called numdevices times
    int tpb = 256;
    int ntblk = BUF_SIZ / tpb;
    if (BUF_SIZ % ntblk != 0)
        ntblk++;
    bumparray<<<ntblk, tpb>>>(g[0], BUF_SIZ);

    // go around the ring and copy and bump
    for (int i = 1; i < numDevices; i++)
    {
        cudaSetDevice(i);
        cudaMalloc(&g[i], BUF_SIZ * sizeof(int));
        cudaMemset(g[i], 0, BUF_SIZ * sizeof(int));
        cudaDeviceEnablePeerAccess(i - 1, 0);
        cudaMemcpyPeer(g[i], i, g[i - 1], i - 1, BUF_SIZ * sizeof(int));
        bumparray<<<ntblk, tpb>>>(g[i], BUF_SIZ);
        // Synchronize here if necessary, depending on your application
        cudaDeviceSynchronize();
    }

    // now, copy the result from the final device to the host
    cudaMemcpy(h_copy, g[numDevices - 1], BUF_SIZ * sizeof(int), cudaMemcpyDeviceToHost);

    // now compare the results
    // ... modify the original and compare
    for (int i = 0; i < BUF_SIZ; i++)
    {
        h[i] += numDevices;
        if (h_copy[i] - h[i] != 0)
        {
            std::cout << "ERROR in the copying or bump routines" << std::endl;
            std::cout << "\th: " << h[i] << " h_copy: " << h_copy[i] << std::endl;
            break;
        }
    }

    // Clean up memory and disable peer access
    cudaSetDevice(0);
    cudaFree(g[0]);
    for (int i = 1; i < numDevices; i++)
    {
        cudaSetDevice(i);
        cudaDeviceDisablePeerAccess(i - 1);
        cudaFree(g[i]);
    }

    // Free the arrays of pointers
    delete h;
    delete h_copy;
    delete[] g;

    return 0;
}
