#include <iostream>
#include <cuda_runtime.h>

__global__ void sumWithSharedMemory(int *input, int *output, int n, int sharedDataSize)
{
    extern __shared__ int sharedData[];

    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    // Load data from global memory to shared memory
    if (globalIdx < n)
        sharedData[localIdx] = input[globalIdx];
    else
        sharedData[localIdx] = 0;

    __syncthreads();

    // Perform the sum reduction using shared memory
    for (int stride = sharedDataSize / 2; stride > 0; stride >>= 1)
    {
        if (localIdx < stride && localIdx + stride < sharedDataSize)
        {
            sharedData[localIdx] += sharedData[localIdx + stride];
        }
        __syncthreads();
    }

    // Write the result back to global memory
    if (localIdx == 0)
    {
        output[blockIdx.x] = sharedData[0];
    }
}

void sumWithHost(int *input, int *output, int n)
{
    for (int i = 0; i < n; i++)
    {
        output[0] += input[i];
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        std::cout << "Usage: ./program <N> <THREADS_PER_BLOCK> <SHARED_DATA_SIZE>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int THREADS_PER_BLOCK = std::atoi(argv[2]);
    int SHARED_DATA_SIZE = std::atoi(argv[3]);

    int *input = new int[N];
    int *output = new int[N / THREADS_PER_BLOCK];
    int *gpuOutput = new int[N / THREADS_PER_BLOCK];

    // Initialize input array
    for (int i = 0; i < N; i++)
    {
        input[i] = i + 1;
    }

    int *d_input;
    int *d_output;

    cudaMalloc((void **)&d_input, sizeof(int) * N);
    cudaMalloc((void **)&d_output, sizeof(int) * (N / THREADS_PER_BLOCK));

    cudaMemcpy(d_input, input, sizeof(int) * N, cudaMemcpyHostToDevice);

    sumWithSharedMemory<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK, sizeof(int) * SHARED_DATA_SIZE>>>(d_input, d_output, N, SHARED_DATA_SIZE);

    cudaMemcpy(gpuOutput, d_output, sizeof(int) * (N / THREADS_PER_BLOCK), cudaMemcpyDeviceToHost);

    sumWithHost(input, output, N);

    std::cout << "GPU Result: ";
    for (int i = 0; i < N / THREADS_PER_BLOCK; i++)
    {
        std::cout << gpuOutput[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Host Result: ";
    for (int i = 0; i < N / THREADS_PER_BLOCK; i++)
    {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    int gpuSum = 0;
    for (int i = 0; i < N / THREADS_PER_BLOCK; i++)
    {
        gpuSum += gpuOutput[i];
    }

    if (gpuSum == output[0])
    {
        std::cout << "Result is correct!" << std::endl;
    }
    else
    {
        std::cout << "Result is incorrect!" << std::endl;
    }

    delete[] input;
    delete[] output;
    delete[] gpuOutput;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

