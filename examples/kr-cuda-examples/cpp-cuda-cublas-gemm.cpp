#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 8192
#define M 8192
#define K 8192

int main()
{
    // Allocate host memory for matrices
    float *h_A = new float[N * K];
    float *h_B = new float[K * M];
    float *h_C = new float[N * M];

    // Initialize input matrices
    for (int i = 0; i < N * K; ++i)
        h_A[i] = static_cast<float>(i % 100) / 100.0f;

    for (int i = 0; i < K * M; ++i)
        h_B[i] = static_cast<float>((i + 1) % 100) / 100.0f;

    // Allocate device memory for matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, N * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * M * sizeof(float));
    cudaMalloc((void **)&d_C, N * M * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * M * sizeof(float), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication
    const float alpha = 1.0f;
    const float beta = -0.5f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_A, N, d_B, K, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print performance information
    std::cout << "Matrix multiplication performance:" << std::endl;
    std::cout << "Matrix size: " << N << "x" << M << "x" << K << std::endl;
    std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;
    float flop = 2. * (static_cast<float>(N) * static_cast<float>(N) * static_cast<float>(N) + static_cast<float>(N) * static_cast<float>(N));
    // flop / time[s]
    std::cout << "(flop: " << flop << ") GFLOPs: " << flop * 1000.0f / milliseconds / 1.e9 << std::endl;

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
