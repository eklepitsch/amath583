#ifndef REF_GEMVT_HPP
#define REF_GEMVT_HPP

#include <iostream>
#include <vector>

template<typename T>
bool bounds_check(const std::vector<std::vector<T>>& A,
                  const std::vector<T> &x,
                  std::vector<T> &y)
{
    if(A.empty()) {std::cout << "Empty A" << std::endl; return false;}
    if(A[0].empty()) {std::cout << "Null row size" << std::endl; return false;}
    auto m = A.size();
    auto n = A[0].size();
    if(n != x.size()) {std::cout << "Invalid x size" << std::endl; return false;}
    if(m != y.size()) {std::cout << "Invalid y size" << std::endl; return false;}
    for(auto i : A)
    {
        // All rows of A must be the same size.
        if(i.size() != n) {std::cout << "Invalid row size" << std::endl; return false;}
    }
    return true;
}

template<typename T>
void gemv(T a,
          const std::vector<std::vector<T>>& A,
          const std::vector<T> &x,
          T b,
          std::vector<T> &y)
{
    if(!bounds_check(A, x, y)) {return;}

    auto m = A.size();
    auto n = A[0].size();

    for(auto i=0; i<m; ++i)
    {
        y[i] *= b;
        for(auto j=0; j<n; ++j)
        {
            y[i] += a * A[i][j] * x[j]; 
        }
    }
}

#endif // REF_GEMVT_HPP