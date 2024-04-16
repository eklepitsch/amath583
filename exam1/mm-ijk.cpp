#include <iostream>
#include "mm-ijk.hpp"

template<typename T>
void mm_ijk(T a,
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

   // Multiply A and B
   std::vector<T> AB;
   AB.resize(m*n);

   for(auto i=0; i<m; ++i)
   {
      for(auto j=0; j<n; ++j)
      {
         for(auto k=0; k<p; ++k)
         {

         }
      }
   }
}

template void mm_ijk<double>(double a,
                             const std::vector<double>& A,
                             const std::vector<double>& B,
                             double b,
                             std::vector<double>& C,
                             int m, int p, int n);

template void mm_ijk<float>(float a,
                            const std::vector<float>& A,
                            const std::vector<float>& B,
                            float b,
                            std::vector<float>& C,
                            int m, int p, int n);

template void mm_ijk<int>(int a,
                          const std::vector<int>& A,
                          const std::vector<int>& B,
                          int b,
                          std::vector<int>& C,
                          int m, int p, int n);

