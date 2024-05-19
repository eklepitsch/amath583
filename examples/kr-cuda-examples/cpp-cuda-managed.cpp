#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void incrementArray(int *array, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
        array[tid]++;
}

template <int tvar>
__global__ void gpufnc(int *array, int size)
{

    int gidx = threadIdx.x + blockIdx.x * blockDim.x;
    switch (tvar)
    {
    case 0:
        if (gidx < size)
            array[gidx] = gidx;
        break;

    case 1:
        if (gidx < size)
            array[gidx] = gidx * 2;
        break;

    default:
        if (gidx < size)
            array[gidx] = 0;
    }
}

int main()
{
    const int size = 10;
    int *array;

    cudaMallocManaged(&array, size * sizeof(int));
    for (int i = 0; i < size; i++)
        array[i] = i;

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    incrementArray<<<numBlocks, blockSize>>>(array, size);
    cudaDeviceSynchronize(); // is this needed

    for (int i = 0; i < size; i++)
        std::cout << "array[" << i << "] = " << array[i] << std::endl;

    int *harray = new int[size];
    int *darray;
    cudaMalloc((void **)&darray, sizeof(int) * size); // get some gpu memory

    gpufnc<0><<<numBlocks, blockSize>>>(darray, size); // mode '0'

    cudaMemcpy(harray, darray, sizeof(int) * size, cudaMemcpyDeviceToHost);
    std::cout << "output from mode 0:" << std::endl;
    for (int i = 0; i < size; i++)
        std::cout << "array[" << i << "] = " << harray[i] << std::endl;

    gpufnc<1><<<numBlocks, blockSize>>>(darray, size); // mode '0'

    cudaMemcpy(harray, darray, sizeof(int) * size, cudaMemcpyDeviceToHost);
    std::cout << "output from mode 1:" << std::endl;
    for (int i = 0; i < size; i++)
        std::cout << "array[" << i << "] = " << harray[i] << std::endl;

    gpufnc<2><<<numBlocks, blockSize>>>(darray, size); // mode '0'

    cudaMemcpy(harray, darray, sizeof(int) * size, cudaMemcpyDeviceToHost);
    std::cout << "output from mode 2:" << std::endl;
    for (int i = 0; i < size; i++)
        std::cout << "array[" << i << "] = " << harray[i] << std::endl;

    delete[] harray;
    cudaFree(array);

    return 0;
}
