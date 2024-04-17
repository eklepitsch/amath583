#include "matrix_swaps.hpp"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <utility>

// [ 1, 2, 3, 4 ]
// [ 3, 4, 7, 8 ]
// [ 9, 0, 1, 2 ]
static const std::vector<double> B = {1, 3, 9, 2, 4, 0, 3, 7, 1, 4, 8, 2};

template<typename T>
void print_matrix(const std::vector<T>& matrix, int m, int n)
{
   for(auto i=0; i<m; ++i)
   {
      std::cout << "[ ";
      for(auto j=0; j<n; ++j)
      {
         auto k = j*m+i;
         std::cout << std::fixed << std::setprecision(1) << matrix[k] << " ";
      }
      std::cout << "]" << std::endl;
   }
}

std::vector<double> GenerateSquareMatrix(unsigned n)
{
   std::vector<double> matrix;
   matrix.reserve(n*n);
   for(auto i=0; i < n*n; ++i) matrix.push_back((double)(std::rand() % 100));
   return matrix;
}

// For std::pair
std::pair<int, int> GetRandomIndices(int n)
{
   int i = std::rand() % n;
   int j = std::rand() % (n - 1);
   if (j >= i)
   {
      j++;
   }
   return std::make_pair(i, j);
}

int main()
{
   auto m = 3;
   auto n = 4;
   auto mB = B;  // "Mutable" B
   std::cout << "Before swap:" << std::endl;
   print_matrix<double>(mB, m, n);
   swapRows(mB, 3, 4, 0, 2);
   std::cout << "After swap:" << std::endl;
   print_matrix<double>(mB, m, n);

   mB = B;  // "Mutable" B
   std::cout << "Before swap:" << std::endl;
   print_matrix<double>(mB, m, n);
   swapCols(mB, 3, 4, 0, 2);
   std::cout << "After swap:" << std::endl;
   print_matrix<double>(mB, m, n);

   auto start = std::chrono::high_resolution_clock::now();
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
   long double elapsedtime = 0.L;
   long double avgtime;
   const unsigned ntrials = 5;
   for(unsigned p=4; p<13; ++p)
   {
      unsigned n = 0x1 << p;
      std::cout << "n: " << n << std::endl;
      auto M = GenerateSquareMatrix(n);

      for(auto i=0; i<ntrials; ++i)
      {
         auto swapIndices = GetRandomIndices(n);

         start = std::chrono::high_resolution_clock::now();
         swapRows(M, n, n, swapIndices.first, swapIndices.second);
         stop = std::chrono::high_resolution_clock::now();
         duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
         elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
      }
      avgtime = elapsedtime/static_cast<long double>(ntrials);
      std::cout << std::setprecision(10) << "Row swap avg time: " << avgtime << std::endl;
      elapsedtime=0.L;

      for(auto i=0; i<ntrials; ++i)
      {
         auto swapIndices = GetRandomIndices(n);

         start = std::chrono::high_resolution_clock::now();
         swapCols(M, n, n, swapIndices.first, swapIndices.second);
         stop = std::chrono::high_resolution_clock::now();
         duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
         elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
      }
      avgtime = elapsedtime/static_cast<long double>(ntrials);
      std::cout << std::setprecision(10) << "Col swap avg time: " << avgtime << std::endl;
      elapsedtime=0.L;
   }
}