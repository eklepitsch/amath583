#include <iostream>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

// Function to compute on CUDA device
template <int functionId>
__global__ void computeFunction(float *data, float *results, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        // Compute different functions based on functionId
        switch (functionId)
        {
        case 0:
            // Function 0: Square the value
            results[tid] = data[tid] * data[tid];
            break;
        case 1:
            // Function 1: Take the square root
            if (data[tid] >= 0.0f)
            {
                results[tid] = sqrtf(data[tid]);
            }
            break;
        case 2:
            // Function 2: Compute the reciprocal
            results[tid] = 1.0f / data[tid];
            break;
        default:
            break;
        }
    }
}

int main()
{
    const int dataSize = 1024;
    const int numFunctions = 3;

    // Allocate and initialize input data on the host
    float *hostData = new float[dataSize];
    for (int i = 0; i < dataSize; i++)
    {
        hostData[i] = static_cast<float>(i + 1);
    }

    // Allocate memory on the device for input data
    float *deviceData;
    cudaMalloc((void **)&deviceData, dataSize * sizeof(float));

    // Allocate memory on the device for intermediate and final result buffers
    float *deviceResults[numFunctions];
    for (int i = 0; i < numFunctions; i++)
    {
        cudaMalloc((void **)&deviceResults[i], dataSize * sizeof(float));
    }

    // Create CUDA streams
    cudaStream_t streams[numFunctions];
    for (int i = 0; i < numFunctions; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Create CUDA events for synchronization
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Allocate host buffer for result data from each stream
    float *hostResults = new float[numFunctions * dataSize];

    // Copy input data from host to device
    cudaMemcpy(deviceData, hostData, dataSize * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel asynchronously on different streams
    for (int i = 0; i < numFunctions; i++)
    {
        cudaEventRecord(startEvent, streams[i]);
        switch (i)
        {
        case 0:
            computeFunction<0><<<(dataSize + 255) / 256, 256, 0, streams[i]>>>(deviceData, deviceResults[i], dataSize);
            break;

        case 1:
            computeFunction<1><<<(dataSize + 255) / 256, 256, 0, streams[i]>>>(deviceData, deviceResults[i], dataSize);
            break;

        case 2:
            computeFunction<2><<<(dataSize + 255) / 256, 256, 0, streams[i]>>>(deviceData, deviceResults[i], dataSize);
            break;
        }
        cudaEventRecord(stopEvent, streams[i]);
    }

    // Wait for all streams to finish their tasks
    cudaDeviceSynchronize();

    // Copy the results from device to host buffer for each stream
    for (int i = 0; i < numFunctions; i++)
    {
        cudaMemcpyAsync(&hostResults[i * dataSize], deviceResults[i], dataSize * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Wait for all streams to complete the memory copy
    for (int i = 0; i < numFunctions; i++)
    {
        cudaStreamSynchronize(streams[i]);
    }

    // Print the results from each stream
    for (int i = 0; i < numFunctions; i++)
    {
        std::cout << "Results from Stream " << i << ":" << std::endl;
        for (int j = 0; j < dataSize; j++)
        {
            std::cout << hostResults[i * dataSize + j] << " ";
        }
        std::cout << std::endl;
    }

    // Compute and display elapsed time for each stream
    for (int i = 0; i < numFunctions; i++)
    {
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
        std::cout << "Stream " << i << " Elapsed Time: " << elapsedTime << "ms" << std::endl;
    }

    // Free memory
    delete[] hostData;
    delete[] hostResults;
    cudaFree(deviceData);
    for (int i = 0; i < numFunctions; i++)
    {
        cudaFree(deviceResults[i]);
    }

    // Destroy CUDA events and streams
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    for (int i = 0; i < numFunctions; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
