#include <iomanip>
#include <iostream>
#include <vector>
#include <utility>

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

template<typename T>
std::unique_ptr<std::vector<T>> GenerateSquareMatrix(std::size_t n)
{
   auto matrix = std::make_unique<std::vector<T>>();
   matrix->reserve(n*n);
   for(auto i=0; i < n*n; ++i) matrix->push_back((T)(std::rand() % 100));
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