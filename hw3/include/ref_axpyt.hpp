#ifndef REF_AXPYT_HPP
#define REF_AXPYT_HPP

#include <iostream>
#include <vector>

template<typename T>
void axpy(T a,
          const std::vector<T> &x,
          std::vector<T> &y)
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

#endif // REF_AXPYT_HPP

