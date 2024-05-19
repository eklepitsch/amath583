#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ int da;
__constant__ int db;

__global__ void AplusB(int *ret, int i1, int i2)
{
    if (threadIdx.x < 10)
        ret[threadIdx.x + blockIdx.x * blockDim.x] = i1 + i2 + threadIdx.x;
}

int main()
{
    int *ret;
    cudaMallocManaged(&ret, 10 * sizeof(int));
    for (int i = 0; i < 10; i++)
        ret[i] = 0;

    int a = 10;
    int b = 100;
    cudaMemcpyToSymbol(da, &a, sizeof(a));
    cudaMemcpyToSymbol(db, &b, sizeof(b));

    AplusB<<<1, 10>>>(ret, a, b);
    cudaDeviceSynchronize();
    for (int i = 0; i < 10; i++)
        std::cout << i << ": A+B = " << ret[i] << std::endl;

    int ca, cb;

    cudaMemcpyFromSymbol(&ca, da, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&cb, db, sizeof(int), 0, cudaMemcpyDeviceToHost);
    std::cout << "copied a, b: " << ca << " " << cb << std::endl;

    int *addrA;
    int *addrB;
    cudaGetSymbolAddress((void **)&addrA, da);
    cudaGetSymbolAddress((void **)&addrB, db);

    cudaFree(ret);
    return 0;
}

