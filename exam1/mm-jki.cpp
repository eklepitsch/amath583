#include <iostream>
#include "mm-jki.hpp"

template<typename T>
void mm_jki(T a,
            const std::vector<T>& A,
            const std::vector<T>& B,
            T b,
            std::vector<T>& C,
            int m, int p, int n)
{

}

template void mm_jki<double>(double a,
                             const std::vector<double>& A,
                             const std::vector<double>& B,
                             double b,
                             std::vector<double>& C,
                             int m, int p, int n);

template void mm_jki<float>(float a,
                             const std::vector<float>& A,
                             const std::vector<float>& B,
                             float b,
                             std::vector<float>& C,
                             int m, int p, int n);

template void mm_jki<int>(int a,
                             const std::vector<int>& A,
                             const std::vector<int>& B,
                             int b,
                             std::vector<int>& C,
                             int m, int p, int n);

