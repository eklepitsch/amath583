#include <iostream>
#include "mm-kji.hpp"

template<typename T>
void mm_kji(T a,
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
   for(auto k=0; k<p; ++k)
   {
      for(auto j=0; j<n; ++j)
      {
         auto kj = j*p+k;
         for(auto i=0; i<m; ++i)
         {
            auto ij = j*m+i;
            auto ik = k*m+i;
            if(k == 0) C[ij] *= b; // First time through this column (of C)
            C[ij] += a*A[ik]*B[kj]; 
         }
      }
   }
}

template void mm_kji<double>(double a,
                             const std::vector<double>& A,
                             const std::vector<double>& B,
                             double b,
                             std::vector<double>& C,
                             int m, int p, int n);

template void mm_kji<float>(float a,
                             const std::vector<float>& A,
                             const std::vector<float>& B,
                             float b,
                             std::vector<float>& C,
                             int m, int p, int n);

template void mm_kji<int>(int a,
                             const std::vector<int>& A,
                             const std::vector<int>& B,
                             int b,
                             std::vector<int>& C,
                             int m, int p, int n);

