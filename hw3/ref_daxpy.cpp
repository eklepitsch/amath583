#include "ref_daxpy.hpp"
#include <iostream>

void daxpy(double a, const std::vector<double> &x, std::vector<double> &y)
{
    auto nx = x.size();
    auto ny = y.size();

    if(nx != ny) {std::cout << "invalid x,y size" << std::endl; return;}

    for(auto i=0; i<nx; ++i)
    {
        y[i] += a * x[i];
    }
    return;
}