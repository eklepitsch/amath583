#ifndef REF_GEMVT_HPP
#define REF_GEMVT_HPP

#include <iostream>
#include <vector>

template<typename T>
void gemv(T a,
          const std::vector<std::vector<T>>& A,
          const std::vector<T> &x,
          T b,
          std::vector<T> &y)
{
    auto m = A.size();
    auto n = A[0].size();

    if(n != x.size()) {std::cout << "Invalid x size" << std::endl; return;}
    if(m != y.size()) {std::cout << "Invalid y size" << std::endl; return;}

    for(auto i=0; i<m; ++i)
    {
        // All rows must be the same length as the first row
        if(n != A[i].size()) {std::cout << "Invalid row size" << std::endl; return;}

        y[i] *= b;
        for(auto j=0; j<n; ++j)
        {
            y[i] += a * A[i][j] * x[j]; 
        }
    }
}

#endif // REF_GEMVT_HPP