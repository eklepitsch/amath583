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

typedef std::vector<std::vector<float>> vMatrix;

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

template<typename T>
void assert_matrices_are_equal(const Matrix<T>& a, const Matrix<T>& b)
{
  ASSERT_EQ(a.numRows(), b.numRows());
  ASSERT_EQ(a.numCols(), b.numCols());

  for(auto i=0; i< a.numRows(); ++i)
  {
    for(auto j=0; j< a.numCols(); ++j)
    {
      ASSERT_EQ(a(i, j), b(i, j));
    }
  }
}

TEST(MatrixTest, SimpleConstruction)
{
  auto m = Matrix<float>(2, 2);
  cout << "Empty 2x2 matrix:" << endl;
  print_matrix<float>(m);
  ASSERT_EQ(m(0, 0), 0);
  ASSERT_EQ(m(0, 1), 0);
  ASSERT_EQ(m(1, 0), 0);
  ASSERT_EQ(m(1, 1), 0);

  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;
  cout << "Populated 2x2 matrix:" << endl;
  print_matrix<float>(m);

  ASSERT_EQ(m(0, 0), 1);
  ASSERT_EQ(m(0, 1), 2);
  ASSERT_EQ(m(1, 0), 3);
  ASSERT_EQ(m(1, 1), 4);
}

TEST(MatrixTest, VectorConstruction)
{
  std::vector<std::vector<float>> matrix = {{1,  2,  3,  4},
                                            {5,  6,  7,  8},
                                            {9, 10, 11, 12}};  
  auto m = construct_matrix(matrix);
  cout << "Vector constructed matrix:" << endl;
  print_matrix(m);

  int i = 0;
  for(auto row : matrix)
  {
    int j = 0;
    for(auto value : row)
    {
      ASSERT_EQ(value, m(i, j));
      j++;
    }
    i++;
  }
}

TEST(MatrixTest, Transpose1)
{
  vMatrix a = {{2, 2, 2, 2, 2},
               {0, 0, 0, 0, 0}};
  auto A = construct_matrix(a);
  vMatrix at = {{2, 0},
                {2, 0},
                {2, 0},
                {2, 0},
                {2, 0}};
  auto AT = construct_matrix(at);
  auto result = A.transpose();
  cout << "The matrix A:" << endl;
  print_matrix(A);
  cout << "The matrix A, transposed:" << endl;
  print_matrix(result);
  assert_matrices_are_equal(result, AT);
}

TEST(MatrixTest, Transpose2)
{
  vMatrix a = {{1, 2, 1, 2, 1},
               {3, 4, 3, 4, 3},
               {5, 6, 5, 6, 5},
               {7, 8, 7, 8, 7},
               {2, 1, 2, 1, 2},
               {1, 1, 1, 1, 1},};
  auto A = construct_matrix(a);
  vMatrix at = {{1, 3, 5, 7, 2, 1},
                {2, 4, 6, 8, 1, 1},
                {1, 3, 5, 7, 2, 1},
                {2, 4, 6, 8, 1, 1},
                {1, 3, 5, 7, 2, 1}};
  auto AT = construct_matrix(at);
  auto result = A.transpose();
  cout << "The matrix A:" << endl;
  print_matrix(A);
  cout << "The matrix A, transposed:" << endl;
  print_matrix(result);
  assert_matrices_are_equal(result, AT);
}
