// starter code for final exam problem 1

#ifndef MYSPARSE_HPP
#define MYSPARSE_HPP

#include <vector>
#include <utility> // for std::pair
#include <algorithm>  // for std::set_intersection
#include <execution>  // for std::execution::seq

using namespace std;

// Template class for generic sparse vector representation
template <class T>
class mysparse
{
public:
    vector<pair<int, T>> sp_idx_val;
    int size;

    // constructor to convert dense vector to sparse representation
    mysparse(const vector<T> &dvec)
    {
        for(auto i = 0; i<dvec.size(); i++)
        {
            if(0 != dvec[i])
            {
                sp_idx_val.push_back(make_pair(i, dvec[i]));
            }
        }
        size = dvec.size();
    }

    // method to calculate dot product of two generic sparse vectors
    T spgendot(const mysparse &svec) const
    {
        T result = 0;
        int i = 0, j = 0;

        if(size != svec.size)
        {
            throw std::invalid_argument("Sparse vectors are of different sizes");
        }

        // The constructor guarantees that the indices are sorted, so no need to
        // sort the vectors here (we couldn't anyway, because this function is const).

        while(i < size && j < size)
        {
            // If the indices are the same, these elements contribute to the dot product.
            if(sp_idx_val[i].first == svec.sp_idx_val[j].first)
            {
                result += sp_idx_val[i].second * svec.sp_idx_val[j].second;
                i++;
                j++;
            }
            // If the indices are different, bump one of the indices.
            else if(sp_idx_val[i].first < svec.sp_idx_val[j].first)
            {
                i++;
            }
            else
            {
                j++;
            }
        }
        return result;
    }
};

#endif // MYSPARSE_HPP

