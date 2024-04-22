#include "ref_dgemm.hpp"

#include <iostream>
#include <tuple>

std::tuple<std::size_t, std::size_t, std::size_t> get_mpn(
    const std::vector<std::vector<double>>& A,
    const std::vector<std::vector<double>>& B,
    std::vector<std::vector<double>>& C)
{
    auto m = A.size();
    auto p = A[0].size();
    auto n = B[0].size();
    return {m, p, n};
}

bool bounds_check(const std::vector<std::vector<double>>& A,
                  const std::vector<std::vector<double>>& B,
                  std::vector<std::vector<double>>& C)
{
    auto mpn = get_mpn(A, B, C);
    auto m = std::get<0>(mpn);
    auto p = std::get<1>(mpn);
    auto n = std::get<2>(mpn);

    // A elem R^(mxp)
    if(A.size() != m) {std::cout << "Invalid A col dim" << std::endl; return false;}
    for(auto i : A)
    {
        if(i.size() != p) {std::cout << "Invalid A row dim" << std::endl; return false;}
    }
    // B elem R^(pxn)
    if(B.size() != p) {std::cout << "Invalid B col dim" << std::endl; return false;}
    for(auto i : B)
    {
        if(i.size() != n) {std::cout << "Invalid B row dim" << std::endl; return false;}
    }
    // C elem R^(mxn)
    if(C.size() != m) {std::cout << "Invalid C col dim" << std::endl; return false;}
    for(auto i : C)
    {
        if(i.size() != n) {std::cout << "Invalid C row dim" << std::endl; return false;}
    }
    return true;
}

void dgemm(double a,
           const std::vector<std::vector<double>>& A,
           const std::vector<std::vector<double>>& B,
           double b,
           std::vector<std::vector<double>>& C)
{
    if(!bounds_check(A, B, C)) {return;}

    auto mpn = get_mpn(A, B, C);
    auto m = std::get<0>(mpn);
    auto p = std::get<1>(mpn);
    auto n = std::get<2>(mpn);

    for(auto i=0; i<m; ++i)
    {
        for(auto j=0; j<n; ++j)
        {
            C[i][j] *= b;
            for(auto k=0; k<p; ++k)
            {
                C[i][j] += a * A[i][k] * B[k][j];
            }
        }
    }
}