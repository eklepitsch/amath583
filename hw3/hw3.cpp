#include "matrix-utils.hpp"
#include "mm-jki.hpp"
#include "mm-kij.hpp"
#include "ref_daxpy.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

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

int main()
{
   std::cout << "HW 3" << std::endl;
   //problem_1();
   problem_2();
   return 0;
}