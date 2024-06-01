// mysparse:
// implement a C++ template class to convert a dense vector to sparse vector of any numeric type
// implement member function T spgendot(const mysparse &svec) const {}
// compiling:
// g++ -std=c++14 -c -I./ mysparse.cpp ; g++ -o xmysparse mysparse.o
// running:
// ./xmysparse

#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <utility> // for pair

using namespace std;

#include "mysparse.hpp"

int main()
{
    // Sparse vector dot product - double
    vector<double> dense_A = {0., 1., 0., 0., 2.};
    vector<double> dense_B = {0., 0., 0., 0., 1.5};

    mysparse<double> myspv1(dense_A);
    mysparse<double> myspv2(dense_B);
    double dresult = myspv1.spgendot(myspv2);
    cout << "Generic sparse double  vector dot product: " << dresult << endl;

    // Sparse vector dot product - integer
    vector<int> dense_iA = {0, 1, 0, 0, 2, 0, 1, 0, 0, 2};
    vector<int> dense_iB = {1, 0, 0, 2, 1, 0, 0, 0, 2, 1};

    mysparse<int> myspiv1(dense_iA);
    mysparse<int> myspiv2(dense_iB);
    int iresult = myspiv1.spgendot(myspiv2);
    cout << "Generic sparse integer vector dot product: " << iresult << endl;

    return 0;
}
