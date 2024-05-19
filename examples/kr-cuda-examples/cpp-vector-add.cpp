#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void vectorAdditionGPU(int* a, int* b, int* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

void vectorAdditionCPU(int* a, int* b, int* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int size = 1000;
    int* a = new int[size];
    int* b = new int[size];
    int* c_gpu = new int[size];
    int* c_cpu = new int[size];

    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = i;
    }

    int* dev_a;
    int* dev_b;
    int* dev_c;
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    vectorAdditionGPU<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, size);

    cudaMemcpy(c_gpu, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    vectorAdditionCPU(a, b, c_cpu, size);

    bool resultsMatch = true;
    for (int i = 0; i < size; ++i) {
        if (c_gpu[i] != c_cpu[i]) {
            resultsMatch = false;
            break;
        }
    }

    if (resultsMatch) {
        std::cout << "GPU results match CPU results" << std::endl;
    } else {
        std::cout << "GPU results do not match CPU results" << std::endl;
    }

    delete[] a;
    delete[] b;
    delete[] c_gpu;
    delete[] c_cpu;

    return 0;
}
