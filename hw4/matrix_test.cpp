#include "matrix_class.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace std;

//class MatrixTest : public test::Test
//{
//  protected:
//
//  Matrix<float> mEmpty2x2_;
//  Matrix<float> mA2x2_;
//  Matrix<float> mB2x2_;
//  Matrix<float> mAplusB2x2_;
//};


template<typename T>
void print_matrix(const Matrix<T>& matrix)
{
  for(auto i=0; i<matrix.numRows(); ++i)
  {
    cout << "[ ";
    for(auto j=0; j<matrix.numCols(); ++j)
    {
      cout << matrix(i, j) << " ";
    }
    cout << "]" << endl;
  }
}

template<typename T>
Matrix<T> construct_matrix(const std::vector<std::vector<T>>& matrix)
{
  if(matrix.empty() || matrix[0].empty())
  {
    throw std::invalid_argument("Matrix cannot be empty");
  }
  
  auto nRows = matrix.size();
  auto nCols = matrix[0].size();
  for(auto row : matrix)
  {
    if(row.size() != nCols)
    {
      throw std::invalid_argument("Matrix has inconsistent column size");
    }
  }

  Matrix<T> m(nRows, nCols);
  int i = 0;
  for(auto row : matrix)
  {
    int j = 0;
    for(auto value : row)
    {
      m(i, j) = value;
      ++j;
    }
    ++i;
  }
  return m;
}

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(MatrixTest, SimpleConstruction)
{
  auto m = Matrix<float>(2, 2);
  cout << "Empty 2x2 matrix:" << endl;
  print_matrix<float>(m);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;
  cout << "Populated 2x2 matrix:" << endl;
  print_matrix<float>(m);
}

TEST(MatrixTest, VectorConstruction)
{
  std::vector<std::vector<float>> matrix = {{1,  2,  3,  4},
                                            {5,  6,  7,  8},
                                            {9, 10, 11, 12}};  
  auto m = construct_matrix(matrix);
  cout << "Vector constructed matrix:" << endl;
  print_matrix(m);
}
