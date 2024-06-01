// starter code for final exam problem 1

#ifndef MYSPARSE_HPP
#define MYSPARSE_HPP

#include <vector>
#include <utility> // for std::pair

using namespace std;

// Template class for generic sparse vector representation
template <class T>
class mysparse
{
public:
    vector<pair<int, T>> sp_idx_val;

    // constructor to convert dense vector to sparse representation
    mysparse(const vector<T> &dvec)
    {
        // implement this code
    }

    // method to calculate dot product of two generic sparse vectors
    T spgendot(const mysparse &svec) const
    {
        T result = 0;
        int i = 0, j = 0;

        // implement the sparse generic dot product using the mysparse data structure
        return result;
    }
};

#endif // MYSPARSE_HPP

