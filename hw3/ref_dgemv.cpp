#include "ref_dgemv.hpp"
#include <iostream>

void dgemv(double a,
           const std::vector<std::vector<double>>& A,
           const std::vector<double> &x,
           double b,
           std::vector<double> &y)
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