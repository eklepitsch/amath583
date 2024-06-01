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
    cout << "Generic sparse double vector dot product: " << dresult << endl;
    if( dresult != 3.0 )
    {
        throw std::runtime_error("FAILED! Expected 3.0");
    }

    // Sparse vector dot product - integer
    vector<int> dense_iA = {0, 1, 0, 0, 2, 0, 1, 0, 0, 2};
    vector<int> dense_iB = {1, 0, 0, 2, 1, 0, 0, 0, 2, 1};

    mysparse<int> myspiv1(dense_iA);
    mysparse<int> myspiv2(dense_iB);
    int iresult = myspiv1.spgendot(myspiv2);
    cout << "Generic sparse integer vector dot product: " << iresult << endl;
    if( iresult != 4 )
    {
        throw std::runtime_error("FAILED! Expected 4.0");
    }

    // Test a few more cases
    vector<float> dense_C = {0.3, 0.5, 1.2, 1000, 0};
    vector<float> dense_D = {1, 20.3, 0., -4., 4.5};
    mysparse<float> myspv3(dense_C);
    mysparse<float> myspv4(dense_D);
    float fresult = myspv3.spgendot(myspv4);
    cout << "Generic sparse double vector dot product: " << fresult << endl;
    if( fresult != -3989.55f )
    {
        throw std::runtime_error("FAILED! Expected -3989.55");
    }

    vector<int> dense_E = {0, 1, 5, 0, 3, 1, 3, 0};
    vector<int> dense_F = {0, 5, -3, 2, 6, 0, 0, 0};
    mysparse<int> myspv5(dense_E);
    mysparse<int> myspv6(dense_F);
    iresult = myspv5.spgendot(myspv6);
    cout << "Generic sparse integer vector dot product: " << iresult << endl;
    if( iresult != 8 )
    {
        throw std::runtime_error("FAILED! Expected 8");
    }

    // Test edge cases of vectors with different lengths
    mysparse<float> I({1.0, 2.0});
    mysparse<float> J({34.0, -6.2, 4.3, 9.0, -0.8});
    // Expect this to throw an exception
    cout << "Expecting exception: " << endl;
    try { int result = I.spgendot(J); }
    catch (const std::invalid_argument &e)
    {
        cout << "Caught exception: " << e.what() << endl;
    }
    mysparse<int> M({1, 2, 3, 4});
    mysparse<int> N({1, 2, 3});
    // Expect this to throw an exception
    cout << "Expecting exception: " << endl;
    try { int result = M.spgendot(N); }
    catch (const std::invalid_argument &e)
    {
        cout << "Caught exception: " << e.what() << endl;
    }

    return 0;
}
