#include "matrix-utils.hpp"
#include <cblas.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
   ofstream results("./artifacts/p3.csv");
   results << "n, l1_flops, l1_avg_time, L1 (daxpy), L1 (daxpy), l2_flops,"
              " l2_avg_time, L2 (dgemv), L2 (dgemv), l3_flops, l3_avg_time,"
              " L3 (dgemm), L3 (dgemm)" << endl;

   unsigned long long n = 2;
   while(n <= 4096)
   {
      long double l1_elapsed = 0.0;
      long double l2_elapsed = 0.0;
      long double l3_elapsed = 0.0;
      long double l1_avgtime = 0.0;
      long double l2_avgtime = 0.0;
      long double l3_avgtime = 0.0;

      auto x = GenerateRandomVector<double>(n);
      auto y = GenerateRandomVector<double>(n);
      auto A = GenerateSquareMatrix<double>(n);
      auto B = GenerateSquareMatrix<double>(n);
      auto C = GenerateSquareMatrix<double>(n);

      unsigned ntrials = 5;
      for(auto i=0; i<ntrials; ++i)
      {
         // Level 1 BLAS
         double alpha = 2.0;
         auto start = std::chrono::high_resolution_clock::now();
         cblas_daxpy(n, alpha, x->data(), 1, y->data(), 1);
         auto stop = std::chrono::high_resolution_clock::now();
         l1_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-9;

         // Level 2 BLAS
         start = std::chrono::high_resolution_clock::now();
         cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 2.0, A->data(), n,
                          x->data(), 1, 2.0, y->data(), 1);
         stop = std::chrono::high_resolution_clock::now();
         l2_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-9;

         // Level 3 BLAS
         start = std::chrono::high_resolution_clock::now();
         cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 2.0,
                          A->data(), n, B->data(), n, 2.0, C->data(), n);
         stop = std::chrono::high_resolution_clock::now();
         l3_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-9;
      }

      l1_avgtime = l1_elapsed / ntrials;
      l2_avgtime = l2_elapsed / ntrials;
      l3_avgtime = l3_elapsed / ntrials;

      // daxpy flop count = 2n
      unsigned long l1_flop = 2*n;
      // dgemv flop count = 2nm + 3m, = 2n^2 + 3n when m=n
      unsigned long long l2_flop = 2*n*n + 3*n;
      // dgemm flop count = 2mnk + 3mn, = 2n^3 + 3n^2 when m=n=k
      unsigned long long l3_flop = 2*n*n*n + 3*n*n;

      results << n << ", " << std::setprecision(10)
         << l1_flop << ", "
         << l1_avgtime << ", "
         << (double)l1_flop/l1_avgtime << ", "
         << ((double)l1_flop/l1_avgtime)/1.e6 << ", "
         << l2_flop << ", "
         << l2_avgtime << ", "
         << (double)l2_flop/l2_avgtime << ", "
         << ((double)l2_flop/l2_avgtime)/1.e6 << ", "
         << l3_flop << ", "
         << l3_avgtime << ", "
         << (double)l3_flop/l3_avgtime << ", "
         << ((double)l3_flop/l3_avgtime)/1.e6
         << endl;

      n *= 2;
   }
   results.close();
   return 0;
}