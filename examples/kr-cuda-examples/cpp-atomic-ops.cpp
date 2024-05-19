#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>


__global__ void atomicOperations(int* data)
{
    atomicAdd((int *)(&data[0]), threadIdx.x);
    atomicSub(&data[1], threadIdx.x);
    atomicExch(&data[2], 42);
    atomicMin(&data[3], threadIdx.x);
    atomicMax(&data[4], threadIdx.x);
    atomicInc((unsigned int*)&data[5], threadIdx.x);
    atomicDec((unsigned int*)&data[6], threadIdx.x);
    atomicCAS((unsigned int*)&data[7], 0, threadIdx.x);
}

int main()
{
    const int dataSize = 8;
    int hostData[dataSize] = { 0 };

    int* deviceData;
    cudaMalloc((void**)&deviceData, dataSize * sizeof(int));
    cudaMemcpy(deviceData, hostData, dataSize * sizeof(int), cudaMemcpyHostToDevice);

    atomicOperations<<<1, 32>>>(deviceData);

    cudaMemcpy(hostData, deviceData, dataSize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dataSize; i++)
    {
        std::cout << "Atomic Operation " << i << ": " << hostData[i] << std::endl;
    }

    cudaFree(deviceData);

    return 0;
}

