#include "matrix-utils.hpp"
#include <cblas.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

void measure_openblas_gemm(unsigned long long n, unsigned ntrials, ofstream& results)
{
   long double elapsed = 0.0;
   long double avgtime = 0.0;

   auto A = GenerateSquareMatrix<double>(n);
   auto B = GenerateSquareMatrix<double>(n);
   auto C = GenerateSquareMatrix<double>(n);

   for(auto i=0; i<ntrials; ++i)
   {
      auto start = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 2.0,
                     A->data(), n, B->data(), n, 2.0, C->data(), n);
      auto stop = std::chrono::high_resolution_clock::now();
      elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
            stop - start).count()*1.e-9;
   }
   avgtime = elapsed/ntrials;

   // dgemm flop count = 2mnk + 3mn, = 2n^3 + 3n^2 when m=n=k
   unsigned long long l3_flop = 2*n*n*n + 3*n*n;

   results << n << ", " << std::setprecision(10)
      << l3_flop << ", "
      << avgtime << ", "
      << (double)l3_flop/avgtime << ", "
      << ((double)l3_flop/avgtime)/1.e9
      << endl;

   return;
}

void measure_cublas_gemm(unsigned long long n, unsigned ntrials,
                         ofstream& results, cublasHandle_t& hndl)
{
   // long double elapsed = 0.0;
   float elapsed = 0.0;
   long double avgtime = 0.0;

   // Host-side matrices in memory (stack)
   auto A = GenerateSquareMatrix<double>(n);
   auto B = GenerateSquareMatrix<double>(n);

   // Device-side matrices (in GPU memory)
   double *dA, *dB, *dC;
   cudaMalloc((void **)&dA, n*n*sizeof(double));
   cudaMalloc((void **)&dB, n*n*sizeof(double));
   cudaMalloc((void **)&dC, n*n*sizeof(double));

   // Copy data from CPU to GPU
   cudaMemcpy(dA, A->data(), n*n*sizeof(double), cudaMemcpyHostToDevice);
   cudaMemcpy(dB, B->data(), n*n*sizeof(double), cudaMemcpyHostToDevice);

   for(auto i=0; i<ntrials; ++i)
   {
      double alpha = 2.0;
      double beta = 2.0;

      // auto start = std::chrono::high_resolution_clock::now();
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
      cudaEventRecord(start);
      cublasDgemm(hndl, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha,
                  dA, n, dB, n, &beta, dC, n);
      // auto stop = std::chrono::high_resolution_clock::now();
      // elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
      //       stop - start).count()*1.e-9;
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsed, start, stop);
      elapsed *= 1.e-3;  // Convert milliseconds to seconds
   }
   avgtime = elapsed/ntrials;

   // Deallocate GPU memory
   cudaFree(dA);
   cudaFree(dB);
   cudaFree(dC);

   // dgemm flop count = 2mnk + 3mn, = 2n^3 + 3n^2 when m=n=k
   unsigned long long l3_flop = 2*n*n*n + 3*n*n;

   results << n << ", " << std::setprecision(10)
      << l3_flop << ", "
      << avgtime << ", "
      << (double)l3_flop/avgtime << ", "
      << ((double)l3_flop/avgtime)/1.e9
      << endl;
   
   return;
}

int main(int argc, char** argv)
{
   if(argc != 2)
   {
      cout << "Usage: " << argv[0] << " <max_matrix_dimension>" << endl;
      return 1;
   }
   unsigned long long max_dim = stoull(argv[1]);

   cublasHandle_t hndl;
   auto cublasStatus = cublasCreate(&hndl);
   if(cublasStatus != CUBLAS_STATUS_SUCCESS)
   {
      cout << "CUBLAS initialization failed with status: " << cublasStatus << endl;
      return 1;
   }

   ofstream openblas_results("./artifacts/p4_openblas.csv");
   ofstream cublas_results("./artifacts/p4_cublas.csv");

   openblas_results << "n, l3_flops, l3_avg_time, dgemm (FLOPs),"
                       " dgemm (GFLOPs)" << endl;
   cublas_results << "n, l3_flops, l3_avg_time, dgemm (FLOPs),"
                     " dgemm (GFLOPs)" << endl;

   for(unsigned long long n=16; n<=max_dim; n*=2)
   {
      unsigned ntrials = 5;
      measure_openblas_gemm(n, ntrials, openblas_results);
      measure_cublas_gemm(n, ntrials, cublas_results, hndl);
   }

   auto cublasDestroyStatus = cublasDestroy(hndl);
   if(cublasDestroyStatus != CUBLAS_STATUS_SUCCESS)
   {
      cout << "CUBLAS destroy failed with status: " << cublasDestroyStatus << endl;
   }

   openblas_results.close();
   cublas_results.close();
   return 0;
}
