#include "exam1.problem1.hpp"
#include <iostream>

constexpr int m = 2;
constexpr int n = 4;
constexpr int p = 3;

// A = m x p
// [ 1, 2, 3 ]
// [ 4, 5, 6 ]
static const std::vector<int> A = {1, 4, 2, 5, 3, 6};

// B = p x n
// [ 1, 2, 3, 4 ]
// [ 3, 4, 7, 8 ]
// [ 9, 0, 1, 2 ]
static const std::vector<int> B = {1, 3, 9, 2, 4, 0, 3, 7, 1, 4, 8, 2};

// C = m x n
// [ 1, 2, 3, 4 ]
// [ 5, 6, 7, 8 ]
static const std::vector<int> C = {1, 5, 2, 6, 3, 7, 4, 8};

void print_matrix(const std::vector<int>& matrix, int m, int n)
{
   for(auto i=0; i<m; ++i)
   {
      std::cout << "[ ";
      for(auto j=0; j<n; ++j)
      {
         auto k = j*m+i;
         std::cout << matrix[k] << " ";
      }
      std::cout << "]" << std::endl;
   }
}

void test_case_invalid_A_dims()
{
   const std::vector<int> iA(0, m*p+1);
   mm_ijk(1, iA, B, 1, C, m, p, n);
}

void test_case_invalid_B_dims()
{
   const std::vector<int> iB(0, p*n+1);
   mm_ijk(1, A, iB, 1, C, m, p, n);
}

void test_case_invalid_C_dims()
{
   std::vector<int> iC(0, m*n+1);
   mm_ijk(1, A, B, 1, iC, m, p, n);
}

int main()
{
   std::cout << "The matrix A:" << std::endl;
   print_matrix(A, m, p);
   std::cout << "The matrix B" << std::endl;
   print_matrix(B, p, n);
   std::cout << "The matrix C" << std::endl;
   print_matrix(C, m, n);

   test_cast_invalid_A_dims();
   test_cast_invalid_B_dims();
   test_cast_invalid_C_dims();
}