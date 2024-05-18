#include "matrix-utils.hpp"
#include <cblas.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

void wrap_cblas_daxpy(const int N, const double alpha, const double* X,
                      const int incX, double* Y, const int incY)
{
   cblas_daxpy(N, alpha, X, incX, Y, incY);
}

void wrap_cblas_dgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                      const int M, const int N, const double alpha, const double* A, const int lda,
                      const double* X, const int incX, const double beta, double* Y, const int incY)
{
   cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

void wrap_cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                      const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
                      const double alpha, const double* A, const int lda, const double* B,
                      const int ldb, const double beta, double* C, const int ldc)
{
   cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

int main(int argc, char** argv)
{
   ofstream results("./artifacts/p3.csv");
   results << "n, l1_duration, l2_duration, l3_duration" << endl;

   unsigned n = 2;
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
         wrap_cblas_daxpy(n, alpha, x->data(), 1, y->data(), 1);
         auto stop = std::chrono::high_resolution_clock::now();
         l1_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-9;

         // Level 2 BLAS
         start = std::chrono::high_resolution_clock::now();
         wrap_cblas_dgemv(CblasColMajor, CblasNoTrans, n, n, 2.0, A->data(), n,
                          x->data(), 1, 2.0, y->data(), 1);
         stop = std::chrono::high_resolution_clock::now();
         l2_elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-9;

         // Level 3 BLAS
         start = std::chrono::high_resolution_clock::now();
         wrap_cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 2.0,
                          A->data(), n, B->data(), n, 2.0, C->data(), n);
         stop = std::chrono::high_resolution_clock::now();
         l3_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-9;
      }

      l1_avgtime = l1_elapsed / ntrials;
      l2_avgtime = l2_elapsed / ntrials;
      l3_avgtime = l3_elapsed / ntrials;

      results << n << ", " << std::setprecision(10) << l1_avgtime << ", "
         << l2_avgtime << ", " << l3_avgtime << endl;

      n *= 2;
   }
   results.close();
   return 0;
}