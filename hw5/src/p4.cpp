#include "matrix-utils.hpp"
#include <cblas.h>
#include <cublas.h>
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
      << ((double)l3_flop/avgtime)/1.e6
      << endl;
   
   return;
}

void measure_cublas_gemm(unsigned long long n, unsigned ntrials, ofstream& results)
{
   long double elapsed = 0.0;
   long double avgtime = 0.0;

   auto A = GenerateSquareMatrix<double>(n);
   auto B = GenerateSquareMatrix<double>(n);
   auto C = GenerateSquareMatrix<double>(n);

   for(auto i=0; i<ntrials; ++i)
   {
      auto start = std::chrono::high_resolution_clock::now();
      // cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 2.0,
      //                A->data(), n, B->data(), n, 2.0, C->data(), n);
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
      << ((double)l3_flop/avgtime)/1.e6
      << endl;
   
   return;
}

int main(int argc, char** argv)
{
   ofstream openblas_results("./artifacts/p4_openblas.csv");
   ofstream cublas_results("./artifacts/p4_cublas.csv");

   openblas_results << "n, l3_flops, l3_avg_time, dgemm (FLOPs),"
                       " dgemm (MFLOPs)" << endl;
   cublas_results << "n, l3_flops, l3_avg_time, dgemm (FLOPs),"
                     " dgemm (MFLOPs)" << endl;

   for(unsigned long long n=16; n<=32; n*=2)
   {
      unsigned ntrials = 5;
      measure_openblas_gemm(n, ntrials, openblas_results);
      measure_cublas_gemm(n, ntrials, cublas_results);
   }

   openblas_results.close();
   cublas_results.close();
   return 0;
}