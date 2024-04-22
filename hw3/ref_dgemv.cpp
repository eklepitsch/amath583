#include "ref_dgemv.hpp"
#include <iostream>

bool bounds_check(const std::vector<std::vector<double>>& A,
                  const std::vector<double> &x,
                  std::vector<double> &y)
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

void dgemv(double a,
           const std::vector<std::vector<double>>& A,
           const std::vector<double> &x,
           double b,
           std::vector<double> &y)
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