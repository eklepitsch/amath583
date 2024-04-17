#include <iostream>
#include "mm-jik.hpp"

template<typename T>
void mm_jik(T a,
            const std::vector<T>& A,
            const std::vector<T>& B,
            T b,
            std::vector<T>& C,
            int m, int p, int n)
{
   // Bounds checking
   if(A.size() != m*p){std::cout << "Invalid A dim" << std::endl; return;}
   if(B.size() != p*n){std::cout << "Invalid B dim" << std::endl; return;}
   if(C.size() != m*n){std::cout << "Invalid C dim" << std::endl; return;}

   // Do the math
   for(auto j=0; j<n; ++j)
   {
      for(auto i=0; i<m; ++i)
      {
         auto ij = j*m+i;
         C[ij] *= b;
         for(auto k=0; k<p; ++k)
         {
            auto ik = k*m+i;
            auto kj = j*p+k;
            C[ij] += a*A[ik]*B[kj];
         }
      }
   }
}

template void mm_jik<double>(double a,
                             const std::vector<double>& A,
                             const std::vector<double>& B,
                             double b,
                             std::vector<double>& C,
                             int m, int p, int n);

template void mm_jik<float>(float a,
                             const std::vector<float>& A,
                             const std::vector<float>& B,
                             float b,
                             std::vector<float>& C,
                             int m, int p, int n);

template void mm_jik<int>(int a,
                             const std::vector<int>& A,
                             const std::vector<int>& B,
                             int b,
                             std::vector<int>& C,
                             int m, int p, int n);

