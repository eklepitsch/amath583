#include "matrix-utils.hpp"

#ifdef USE_LIBRARY
   #include "refBLAS.hpp"
#else
   #include "mm-jki.hpp"
   #include "mm-kij.hpp"
   #include "ref_axpyt.hpp"
   #include "ref_daxpy.hpp"
   #include "ref_dgemv.hpp"
   #include "ref_dgemm.hpp"
   #include "ref_gemmt.hpp"
   #include "ref_gemvt.hpp"
#endif

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

#ifndef USE_LIBRARY
template<typename T>
std::pair<long double, long double> multiply_square_matrices(std::size_t n,
                                                             unsigned ntrials)
{
   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   long double jki_elapsedtime = 0.L;
   long double kij_elapsedtime = 0.L;
   long double jki_avgtime, kij_avgtime;

   for(unsigned i=0; i<ntrials; ++i)
   {
      auto A = GenerateSquareMatrix<T>(n);
      auto B = GenerateSquareMatrix<T>(n);
      auto C = GenerateSquareMatrix<T>(n);

      start = std::chrono::high_resolution_clock::now();
      mm_jki(static_cast<T>(1), *A, *B, static_cast<T>(1), *C, n, n, n);
      stop = std::chrono::high_resolution_clock::now();
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      jki_elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds

      start = std::chrono::high_resolution_clock::now();
      mm_kij(static_cast<T>(1), *A, *B, static_cast<T>(1), *C, n, n, n);
      stop = std::chrono::high_resolution_clock::now();
      duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      kij_elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
   }

   jki_avgtime = jki_elapsedtime/static_cast<long double>(ntrials);
   kij_avgtime = kij_elapsedtime/static_cast<long double>(ntrials);
   return std::pair<long double, long double>(jki_avgtime, kij_avgtime);
}

void problem_1()
{
   std::cout << "HW3, Problem 1" << std::endl;
   std::ofstream float_results, double_results;
   float_results.open("float_results.csv");
   double_results.open("double_results.csv");

   for(auto n=2; n<=512; ++n)
   {
      // O(n^3), 3 because we have 2 multiplies and 1 addition
      auto flops = 3 * std::pow(n, 3);
      auto result_float = multiply_square_matrices<float>(n, 5);
      auto result_double = multiply_square_matrices<double>(n, 5);
      // first = jki, second = kij
      float_results << std::fixed << std::setprecision(0) << n << ", " <<
         flops << ", " << std::scientific << std::setprecision(10) <<
         result_float.first << ", " << flops / result_float.first << ", " <<
         result_float.second << ", " << flops / result_float.second <<
         std::endl;
      double_results << std::fixed << std::setprecision(0) << n << ", " <<
         flops << ", " << std::scientific << std::setprecision(10) <<
         result_double.first << ", " << flops / result_double.first << ", " <<
         result_double.second << ", " << flops / result_double.second <<
         std::endl;
   }
}
#endif // USE_LIBRARY

void test_daxpy()
{
   std::cout << "Testing daxpy..." << std::endl;
   double a = 3;
   std::vector<double> x = {90, 59, 63, 26};
   std::vector<double> y = {40, 26, 72, 36};
   std::vector<double> z = {40, 26, 72, 36, 22};
   std::cout << "Expect invalid" << std::endl;
   daxpy(a, x, z);
   std::cout << "Expect pass" << std::endl;
   daxpy(a, x, y);
   std::vector<double> expected = {310, 203, 261, 114};
   for(auto i=0; i<x.size(); ++i)
   {
      assert(x.size() == y.size());
      assert(y[i] == expected[i]);
   }
   std::cout << "PASS" << std::endl;
}

long double measure_daxpy(size_t n, unsigned ntrials)
{
   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   long double elapsedtime = 0.L;
   long double avgtime;

   for(unsigned i=0; i<ntrials; ++i)
   {
      auto x = GenerateRandomVector<double>(n);
      auto y = GenerateRandomVector<double>(n);
      double a = std::rand() % 10;
      start = std::chrono::high_resolution_clock::now();
      daxpy(a, *x, *y);
      stop = std::chrono::high_resolution_clock::now();
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
   }
   avgtime = elapsedtime/static_cast<long double>(ntrials);
   return avgtime;
}

void problem_2()
{
   std::cout << "HW3, Problem 2" << std::endl;
   test_daxpy(); 

   std::ofstream daxpy_results;
   daxpy_results.open("daxpy_results.csv");
   for(auto n=2; n<=512; ++n)
   {
      // O(n), 2 because we have 1 multiply and 1 add
      auto flops = 2 * n;
      auto result = measure_daxpy(n, 5);
      daxpy_results << std::fixed << std::setprecision(0) <<
         n << ", " << flops << ", " <<
         std::scientific << std::setprecision(10) << result << ", " <<
         flops/result << std::endl;
   }
   daxpy_results.close();
}

void test_dgemv()
{
   std::cout << "Testing dgemv..." << std::endl;
   typedef std::vector<std::vector<double>> matrix_t;
   matrix_t A = {{1, 5, 6},
                 {2, 3, 5}};
   std::vector<double> x = {2, 4, 6};
   std::vector<double> y = {1, 3};
   std::vector<double> y_orig = {1, 3};
   std::vector<double> expected = {119, 101};
   double a = 2;
   double b = 3;
   std::cout << "Empty vectors..." << std::endl;
   std::vector<double> x_empty, y_empty;
   matrix_t A_empty;
   dgemv(a, A_empty, x, b, y);
   assert(y == y_orig);
   gemv<double>(a, A, x_empty, b, y);
   assert(y == y_orig);
   gemv<double>(a, A, x, b, y_empty);
   assert(y == y_orig);
   std::cout << "Expect invalid..." << std::endl;
   auto x2 = x; x2.push_back(0);
   dgemv(a, A, x2, b, y);
   std::cout << "Expect invalid..." << std::endl;
   auto y2 = y; y2.push_back(0);
   dgemv(a, A, x, b, y2);
   std::cout << "Expect invalid..." << std::endl;
   auto A2 = A; A2[1].push_back(0);
   dgemv(a, A2, x, b, y);
   std::cout << "Expect pass..." << std::endl;
   dgemv(a, A, x, b, y);
   assert(y.size() == expected.size());
   for(auto i=0; i<y.size(); ++i)
   {
      assert(y[i] == expected[i]);
   }
   std::cout << "PASS" << std::endl;
}

long double measure_dgemv(size_t n, unsigned ntrials)
{
   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   long double elapsedtime = 0.L;
   long double avgtime;
   auto m = n;

   for(unsigned i=0; i<ntrials; ++i)
   {
      double a = std::rand() % 10;
      auto A = GenerateRandomVectorOfVectors<double>(m, n);
      auto x = GenerateRandomVector<double>(n);
      auto b = std::rand() % 10;
      auto y = GenerateRandomVector<double>(m);
      start = std::chrono::high_resolution_clock::now();
      dgemv(a, *A, *x, b, *y);
      stop = std::chrono::high_resolution_clock::now();
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
   }
   avgtime = elapsedtime/static_cast<long double>(ntrials);
   return avgtime;
}

void problem_3()
{
   std::cout << "HW3, Problem 3" << std::endl;
   test_dgemv();

   std::ofstream dgemv_results;
   dgemv_results.open("dgemv_results.csv");
   for(auto n=2; n<=512; ++n)
   {
      // O(n^2), 3 because we have 2 multiplies and 1 add
      auto flops = 3 * n * n;
      auto result = measure_dgemv(n, 5);
      dgemv_results << std::fixed << std::setprecision(0) <<
         n << ", " << flops << ", " <<
         std::scientific << std::setprecision(10) << result << ", " <<
         flops/result << std::endl;
   }
   dgemv_results.close();
}

void test_dgemm()
{
   std::cout << "Testing dgemm..." << std::endl;
   constexpr int m = 2;
   constexpr int n = 4;
   constexpr int p = 3;
   // A = m x p
   // [ 1, 2, 3 ]
   // [ 4, 5, 6 ]
   std::vector<std::vector<double>> A = {{1, 2, 3},
                                         {4, 5, 6}};

   // B = p x n
   // [ 1, 2, 3, 4 ]
   // [ 3, 4, 7, 8 ]
   // [ 9, 0, 1, 2 ]
   std::vector<std::vector<double>> B = {{1, 2, 3, 4},
                                         {3, 4, 7, 8},
                                         {9, 0, 1, 2}};

   // C = m x n
   // [ 1, 2, 3, 4 ]
   // [ 5, 6, 7, 8 ]
   std::vector<std::vector<double>> C = {{1, 2, 3, 4},
                                         {5, 6, 7, 8}};
   std::vector<std::vector<double>> C_orig = {{1, 2, 3, 4},
                                              {5, 6, 7, 8}};

   // Expected result for alpha=1, beta=1
   // [ 35, 12, 23, 30 ]
   // [ 78, 34, 60, 76 ]
   const std::vector<std::vector<double>> R1 = {{35, 12, 23, 30},
                                                {78, 34, 60, 76}};

   // Expected result for alpha=2, beta=3
   // [  71, 26,  49,  64 ]
   // [ 161, 74, 127, 160 ]
   const std::vector<std::vector<double>> R2 = {{71,  26, 49,  64},
                                                {161, 74, 127, 160}};

   std::cout << "Empty vectors..." << std::endl;
   std::vector<std::vector<double>> empty_A, empty_B, empty_C;
   dgemm(1, empty_A, B, 1, C);
   assert(C == C_orig);
   dgemm(1, A, empty_B, 1, C);
   assert(C == C_orig);
   dgemm(1, A, B, 1, empty_C);
   assert(C == C_orig);
   std::cout << "Expect invalid..." << std::endl;
   auto A2 = A; A2.push_back({7, 8, 9});
   dgemm(1, A2, B, 1, C);
   std::cout << "Expect invalid..." << std::endl;
   A2 = A; A2[0].push_back(0);
   dgemm(1, A2, B, 1, C);
   std::cout << "Expect invalid..." << std::endl;
   auto B2 = B; B2.push_back({7, 8, 9, 10});
   dgemm(1, A, B2, 1, C);
   std::cout << "Expect invalid..." << std::endl;
   B2 = B; B2[0].push_back(0);
   dgemm(1, A, B2, 1, C);
   std::cout << "Expect invalid..." << std::endl;
   auto C2 = C; C2.push_back({7, 8, 9, 10});
   dgemm(1, A, B, 1, C2);
   std::cout << "Expect invalid..." << std::endl;
   C2 = C; C2[0].push_back(0);
   dgemm(1, A, B, 1, C2);

   std::cout << "Testing a=1, b=1..." << std::endl;
   auto Cm = C;  // "mutable"
   dgemm(1, A, B, 1, Cm);
   for(auto i=0; i<m; ++i)
   {
      for(auto j=0; j<n; ++j)
      {
         assert(Cm[i][j] == R1[i][j]);
      }
   }

   std::cout << "Testing a=2, b=3..." << std::endl;
   Cm = C;  // "mutable"
   dgemm(2, A, B, 3, Cm);
   for(auto i=0; i<m; ++i)
   {
      for(auto j=0; j<n; ++j)
      {
         assert(Cm[i][j] == R2[i][j]);
      }
   }
   std::cout << "PASS" << std::endl;
}

long double measure_dgemm(std::size_t n, unsigned ntrials)
{
   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   long double elapsedtime = 0.L;
   long double avgtime;

   for(unsigned i=0; i<ntrials; ++i)
   {
      double a = std::rand() % 10;
      double b = std::rand() % 10;
      auto A = GenerateRandomVectorOfVectors<double>(n, n);
      auto B = GenerateRandomVectorOfVectors<double>(n, n);
      auto C = GenerateRandomVectorOfVectors<double>(n, n);
      start = std::chrono::high_resolution_clock::now();
      dgemm(a, *A, *B, b, *C);
      stop = std::chrono::high_resolution_clock::now();
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
   }
   avgtime = elapsedtime/static_cast<long double>(ntrials);
   return avgtime;
}

void problem_4()
{
   std::cout << "HW3, Problem 4" << std::endl;
   test_dgemm();

   std::ofstream dgemm_results;
   dgemm_results.open("dgemm_results.csv");
   for(auto n=2; n<=512; ++n)
   {
      // 3 * O(n^3) + O(n^2)
      auto flops = 3 * std::pow(n, 3) + std::pow(n, 2);
      auto result = measure_dgemm(n, 5);
      dgemm_results << std::fixed << std::setprecision(0) <<
         n << ", " << flops << ", " <<
         std::scientific << std::setprecision(10) << result << ", " <<
         flops/result << std::endl;
   }
   dgemm_results.close();
}

template<typename T>
void test_axpy()
{
   std::cout << "Testing axpy..." << std::endl;
   T a = 3;
   std::vector<T> x = {90, 59, 63, 26};
   std::vector<T> y = {40, 26, 72, 36};
   std::vector<T> z = {40, 26, 72, 36, 22};
   std::cout << "Empty vectors" << std::endl;
   std::vector<T> x_empty, y_empty;
   axpy<T>(a, x_empty, y_empty);
   assert(y_empty.size() == 0);
   std::cout << "Expect invalid" << std::endl;
   axpy<T>(a, x, z);
   std::cout << "Expect pass" << std::endl;
   axpy<T>(a, x, y);
   std::vector<T> expected = {310, 203, 261, 114};
   for(auto i=0; i<x.size(); ++i)
   {
      assert(x.size() == y.size());
      assert(y[i] == expected[i]);
   }
   std::cout << "PASS" << std::endl;
}

template<typename T>
void test_gemv()
{
   std::cout << "Testing gemv..." << std::endl;
   typedef std::vector<std::vector<T>> matrix_t;
   matrix_t A = {{1, 5, 6},
                 {2, 3, 5}};
   std::vector<T> x = {2, 4, 6};
   std::vector<T> y = {1, 3};
   std::vector<T> y_orig = {1, 3};
   std::vector<T> expected = {119, 101};
   T a = 2;
   T b = 3;
   std::cout << "Empty vectors..." << std::endl;
   std::vector<T> x_empty, y_empty;
   matrix_t A_empty;
   gemv<T>(a, A_empty, x, b, y);
   assert(y == y_orig);
   gemv<T>(a, A, x_empty, b, y);
   assert(y == y_orig);
   gemv<T>(a, A, x, b, y_empty);
   assert(y == y_orig);
   std::cout << "Expect invalid..." << std::endl;
   auto x2 = x; x2.push_back(0);
   gemv<T>(a, A, x2, b, y);
   std::cout << "Expect invalid..." << std::endl;
   auto y2 = y; y2.push_back(0);
   gemv<T>(a, A, x, b, y2);
   std::cout << "Expect invalid..." << std::endl;
   auto A2 = A; A2[1].push_back(0);
   gemv<T>(a, A2, x, b, y);
   std::cout << "Expect pass..." << std::endl;
   gemv<T>(a, A, x, b, y);
   assert(y.size() == expected.size());
   for(auto i=0; i<y.size(); ++i)
   {
      assert(y[i] == expected[i]);
   }
   std::cout << "PASS" << std::endl;
}

template<typename T>
void test_gemm()
{
   std::cout << "Testing gemm..." << std::endl;
   constexpr int m = 2;
   constexpr int n = 4;
   constexpr int p = 3;
   // A = m x p
   // [ 1, 2, 3 ]
   // [ 4, 5, 6 ]
   std::vector<std::vector<T>> A = {{1, 2, 3},
                                    {4, 5, 6}};

   // B = p x n
   // [ 1, 2, 3, 4 ]
   // [ 3, 4, 7, 8 ]
   // [ 9, 0, 1, 2 ]
   std::vector<std::vector<T>> B = {{1, 2, 3, 4},
                                    {3, 4, 7, 8},
                                    {9, 0, 1, 2}};

   // C = m x n
   // [ 1, 2, 3, 4 ]
   // [ 5, 6, 7, 8 ]
   std::vector<std::vector<T>> C = {{1, 2, 3, 4},
                                    {5, 6, 7, 8}};
   std::vector<std::vector<T>> C_orig = {{1, 2, 3, 4},
                                         {5, 6, 7, 8}};

   // Expected result for alpha=1, beta=1
   // [ 35, 12, 23, 30 ]
   // [ 78, 34, 60, 76 ]
   const std::vector<std::vector<T>> R1 = {{35, 12, 23, 30},
                                           {78, 34, 60, 76}};

   // Expected result for alpha=2, beta=3
   // [  71, 26,  49,  64 ]
   // [ 161, 74, 127, 160 ]
   const std::vector<std::vector<T>> R2 = {{71,  26, 49,  64},
                                           {161, 74, 127, 160}};

   std::cout << "Empty vectors..." << std::endl;
   std::vector<std::vector<T>> empty_A, empty_B, empty_C;
   gemm<T>(1, empty_A, B, 1, C);
   assert(C == C_orig);
   gemm<T>(1, A, empty_B, 1, C);
   assert(C == C_orig);
   gemm<T>(1, A, B, 1, empty_C);
   assert(C == C_orig);
   std::cout << "Expect invalid..." << std::endl;
   auto A2 = A; A2.push_back({7, 8, 9});
   gemm<T>(1, A2, B, 1, C);
   std::cout << "Expect invalid..." << std::endl;
   A2 = A; A2[0].push_back(0);
   gemm<T>(1, A2, B, 1, C);
   std::cout << "Expect invalid..." << std::endl;
   auto B2 = B; B2.push_back({7, 8, 9, 10});
   gemm<T>(1, A, B2, 1, C);
   std::cout << "Expect invalid..." << std::endl;
   B2 = B; B2[0].push_back(0);
   gemm<T>(1, A, B2, 1, C);
   std::cout << "Expect invalid..." << std::endl;
   auto C2 = C; C2.push_back({7, 8, 9, 10});
   gemm<T>(1, A, B, 1, C2);
   std::cout << "Expect invalid..." << std::endl;
   C2 = C; C2[0].push_back(0);
   gemm<T>(1, A, B, 1, C2);

   std::cout << "Testing a=1, b=1..." << std::endl;
   auto Cm = C;  // "mutable"
   gemm<T>(1, A, B, 1, Cm);
   for(auto i=0; i<m; ++i)
   {
      for(auto j=0; j<n; ++j)
      {
         assert(Cm[i][j] == R1[i][j]);
      }
   }

   std::cout << "Testing a=2, b=3..." << std::endl;
   Cm = C;  // "mutable"
   gemm<T>(2, A, B, 3, Cm);
   for(auto i=0; i<m; ++i)
   {
      for(auto j=0; j<n; ++j)
      {
         assert(Cm[i][j] == R2[i][j]);
      }
   }
   std::cout << "PASS" << std::endl;
}

void problem_5()
{
   std::cout << "HW3, Problem 5" << std::endl;
   test_axpy<int>();
   test_axpy<short>();
   test_axpy<unsigned>();
   test_axpy<float>();
   test_axpy<double>();
   test_axpy<long double>();

   test_gemv<int>();
   test_gemv<short>();
   test_gemv<unsigned>();
   test_gemv<float>();
   test_gemv<double>();
   test_gemv<long double>();

   test_gemm<int>();
   test_gemm<short>();
   test_gemm<unsigned>();
   test_gemm<float>();
   test_gemm<double>();
   test_gemm<long double>();
}

int main()
{
   std::cout << "HW 3" << std::endl;
#ifndef USE_LIBRARY
   problem_1();
#endif
   problem_2();
   problem_3();
   problem_4();
   problem_5();
   return 0;
}