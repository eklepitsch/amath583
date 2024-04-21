#include "matrix-utils.hpp"
#include "mm-jki.hpp"
#include "mm-kij.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

template<typename T>
long double multiply_square_matrices(std::size_t n, unsigned ntrials)
{
   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   long double elapsedtime = 0.L;
   long double avgtime;

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
      elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
   }

   avgtime = elapsedtime/static_cast<long double>(ntrials);
   return avgtime;
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
      float_results << std::fixed << std::setprecision(0) << n << ", " <<
         flops << ", " << std::scientific << std::setprecision(10) <<
         multiply_square_matrices<float>(n, 5) << std::endl;
      double_results << std::fixed << std::setprecision(0) << n << ", " <<
         flops << ", " << std::scientific << std::setprecision(10) <<
         multiply_square_matrices<double>(n, 5) << std::endl;
   }
}

int main()
{
   std::cout << "HW 3" << std::endl;
   problem_1();
   return 0;
}