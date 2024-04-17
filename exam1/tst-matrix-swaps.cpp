#include "matrix_swaps.hpp"
#include <chrono>
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

// For std::pair
std::pair<int, int> getRandomIndices(int n)
{
   int i = std::rand() % n;
   int j = std::rand() % (n - 1);
   if (j >= i)
   {
   j++;
   }
   return std::make_pair(i, j);
}

// ...from inside main()
// std::pair<int, int> rowIndices = getRandomIndices(M);
// int i = rowIndices.first;
// int j = rowIndices.second;
// std::pair<int, int> colIndices = getRandomIndices(N);
// ...

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

   //timer foo
   auto start=std::chrono::high_resolution_clock::now();
   auto stop=std::chrono::high_resolution_clock::now();
   auto duration=std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
   long double elapsedtime=0.L;
   long double avgtime;
   const int ntrials=3;
   //loop on problem size
   for(int i=2;i<=128;i++)
   {
      //perform an experiment
      for(int t=0;t<ntrials;t++)
      {
         start=std::chrono::high_resolution_clock::now();
         //do work(size i, trial t)
         stop=std::chrono::high_resolution_clock::now();
         duration=std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
         elapsedtime+=(duration.count()*1.e-9); //Convert duration to seconds
      }
      avgtime=elapsedtime/static_cast<long double>(ntrials);
      //save or report findings
      //zero time again
      elapsedtime=0.L;
   }
}