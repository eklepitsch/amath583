#include "matrix_class.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace std;

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
               {1, 1, 1, 1, 1}};
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

TEST(MatrixTest, InfinityNorm1)
{
  vMatrix a = {{2, 2, 2, 2, 2},
               {0, 0, 0, 0, 0}};
  auto A = construct_matrix(a);
  auto infinityNorm = A.infinityNorm();
  ASSERT_EQ(infinityNorm, 10);
}

TEST(MatrixTest, InfinityNorm2)
{
  vMatrix a = {{1, 2, 1, 2, 1},
               {3, 4, 3, 4, 3},
               {5, 6, 5, 6, 5},
               {7, 8, 7, 8, 7},
               {2, 1, 2, 1, 2},
               {1, 1, 1, 1, 1}};
  auto A = construct_matrix(a);
  auto infinityNorm = A.infinityNorm();
  ASSERT_EQ(infinityNorm, 37);
}

TEST(MatrixTest, MultiplyInvalidDimensions)
{
  auto a = Matrix<float>(2, 2);
  auto b = Matrix<float>(3, 2);
  EXPECT_ANY_THROW(a * b);
  EXPECT_NO_THROW(b * a);
}

TEST(MatrixTest, Multiply1)
{
  vMatrix a = {{1, 2},
               {3, 4}};
  vMatrix b = {{5, 6, 7},
               {8, 9, 0}};
  vMatrix ab = {{21, 24, 7},
                {47, 54, 21}};

  auto A = construct_matrix(a);
  cout << "The matrix A:" << endl;
  print_matrix(A);
  auto B = construct_matrix(b);
  cout << "The matrix B:" << endl;
  print_matrix(B);
  auto AB = construct_matrix(ab);

  auto result = A * B;
  cout << "The matrix A * B:" << endl;
  print_matrix(result);
  assert_matrices_are_equal(result, AB);
}

TEST(MatrixTest, Multiply2)
{
  vMatrix a = {{1, 2, 1, 2, 1},
               {3, 4, 3, 4, 3},
               {5, 6, 5, 6, 5},
               {7, 8, 7, 8, 7},
               {2, 1, 2, 1, 2},
               {1, 1, 1, 1, 1},
               {7, 8, 7, 8, 7},
               {2, 1, 2, 1, 2}};
  vMatrix b = {{1, 2, 1, 2, 1},
               {3, 4, 3, 4, 3},
               {5, 6, 5, 6, 5},
               {7, 8, 7, 8, 7},
               {2, 1, 2, 1, 2}};
  vMatrix ab = {{28, 33, 28, 33, 28},
                { 64, 75, 64, 75, 64},
                {100, 117, 100, 117, 100},
                {136, 159, 136, 159, 136},
                { 26, 30, 26, 30, 26},
                { 18, 21, 18, 21, 18},
                {136, 159, 136, 159, 136},
                { 26, 30, 26, 30, 26}};

  auto A = construct_matrix(a);
  cout << "The matrix A:" << endl;
  print_matrix(A);
  auto B = construct_matrix(b);
  cout << "The matrix B:" << endl;
  print_matrix(B);
  auto AB = construct_matrix(ab);

  auto result = A * B;
  cout << "The matrix A * B:" << endl;
  print_matrix(result);
  assert_matrices_are_equal(result, AB);
}

TEST(MatrixTest, AddInvalidDimensions)
{
  auto a = Matrix<float>(2, 2);
  auto b = Matrix<float>(3, 2);
  EXPECT_ANY_THROW(a + b);
  EXPECT_ANY_THROW(b + a);
}

TEST(MatrixTest, Add)
{
  vMatrix a = {{3, -3, 9},
               {81, 30, -8}};
  vMatrix b = {{5, 6, 7},
               {8, 9, 0}};
  vMatrix apb = {{8, 3, 16},
                 {89, 39, -8}};

  auto A = construct_matrix(a);
  cout << "The matrix A:" << endl;
  print_matrix(A);
  auto B = construct_matrix(b);
  cout << "The matrix B:" << endl;
  print_matrix(B);
  auto ApB = construct_matrix(apb);

  auto result = A + B;
  cout << "The matrix A + B:" << endl;
  print_matrix(result);
  assert_matrices_are_equal(result, ApB);

  result = B + A;
  cout << "The matrix B + A:" << endl;
  print_matrix(result);
  assert_matrices_are_equal(result, ApB);
}
