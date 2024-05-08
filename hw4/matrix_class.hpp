// AMATH 483-583 row major Matrix class template starter
// write the methods for:
// transpose
// infinityNorm
// operator*
// operator+

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <algorithm>

template <typename T>
class Matrix
{
public:
    Matrix(int numRows, int numCols)
        : num_rows(numRows), num_cols(numCols), data(numRows * numCols) {}

    T &operator()(int i, int j)
    {
        return data[i * num_cols + j];
    }

    const T &operator()(int i, int j) const
    {
        return data[i * num_cols + j];
    }

    Matrix<T> operator*(const Matrix<T> &other) const
    {
        // check for errors in Matrix dimensions
        if(other.num_rows != this->num_cols)
        {
            throw std::invalid_argument("Invalid matrix dimensions");
        }

        auto result_rows = this->num_rows;
        auto result_cols = other.num_cols;
        Matrix<T> result(result_rows, result_cols);
        for(auto i=0; i<result_rows; ++i)
        {
            for(auto j=0; j<result_cols; ++j)
            {
                for(auto k=0; k<this->num_cols; ++k)
                {
                    result.data[i * result_cols + j] +=
                        this->data[i * num_cols + k] *
                        other.data[k * other.num_cols + j];
                }
            }
        }
    }

    Matrix<T> operator+(const Matrix<T> &other) const;

    Matrix<T> transpose() const
    {
        auto result_rows = this->num_cols;
        auto result_cols = this->num_rows;
        Matrix<T> result(result_rows, result_cols);
        for(auto i=0; i<this->num_rows; ++i)
        {
            for(auto j=0; j<this->num_cols; ++j)
            {
                result.data[j * result_cols + i] =
                    this->data[i * this->num_cols + j];
            }
        }
        return result;
    }

    int numRows() const
    {
        return num_rows;
    }

    int numCols() const
    {
        return num_cols;
    }

    T infinityNorm() const
    {
        T norm = 0;
        // Maximum absolute row sum
        for(auto i=0; i<this->num_rows; ++i)
        {
            T sum = 0;
            for(auto j=0; j<this->num_cols; ++j)
            {
                sum += this->data[i * this->num_cols + j];
            }
            if(sum > norm) {norm = sum;}
        }
        
        return norm;
    }

private:
    int num_rows;
    int num_cols;
    std::vector<T> data;
};

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const
{
    // check for errors in Matrix dimensions
    if(other.num_rows != this->num_rows ||
        other.num_cols != this->num_cols)
    {
        throw std::invalid_argument("Invalid matrix dimensions");
    }

    Matrix<T> result(this->num_rows, this->num_cols);
    std::transform(this->data.begin(), this->data.end(),
                   other.data.begin(), result.data.begin(),
                   std::plus<T>());
    return result;
}
