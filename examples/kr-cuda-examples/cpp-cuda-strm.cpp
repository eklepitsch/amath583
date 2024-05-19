#include <iostream>
#include <cuda_runtime.h>

const int numStreams = 4;
const int numThreadsPerStream = 256;
const int numTotalThreads = numStreams * numThreadsPerStream;

__global__ void kernel(int *output, int numThreads, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global thread index

    // Update the corresponding element in the global array
    output[offset + tid] = offset + tid;
}

int main()
{
    int *d_output;                                  // Pointer to device memory
    int outputSize = numTotalThreads * sizeof(int); // Size of the global array in bytes

    // Allocate device memory for the global array
    cudaMalloc((void **)&d_output, outputSize);

    // Create CUDA streams
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; ++i)
        cudaStreamCreate(&streams[i]);

    // Launch kernels on different streams
    for (int i = 0; i < numStreams; ++i)
    {
        // Calculate the offset for each stream
        int offset = i * numThreadsPerStream;

        // Set the CUDA stream for the kernel
        // cudaStream_t stream = streams[i];

        // Launch the kernel
        kernel<<<numThreadsPerStream, 1, 0, streams[i]>>>(d_output, numThreadsPerStream, offset);
    }

    // Synchronize and check for errors on each stream
    for (int i = 0; i < numStreams; ++i)
    {
        cudaStream_t stream = streams[i];
        cudaStreamSynchronize(stream);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
    }

    // Copy the results back to the host
    int *h_output = new int[numTotalThreads];
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < numTotalThreads; ++i)
    {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_output);
    delete[] h_output;
    for (int i = 0; i < numStreams; ++i)
        cudaStreamDestroy(streams[i]);

    return 0;
}
