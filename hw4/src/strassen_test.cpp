#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "strassen.cpp"

using namespace std;

TEST(StrassenTest, InvalidDimensions_NotSquare)
{
   vector<vector<int>> A = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12}};

   vector<vector<int>> B = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12}};

   EXPECT_ANY_THROW(strassenMultiply(A, B));
}

TEST(StrassenTest, InvalidDimensions_NotPowerOfTwo)
{
   vector<vector<int>> A = {{1, 2, 3, 4, 0},
                            {5, 6, 7, 8, 1},
                            {9, 10, 11, 12, 2},
                            {1, 2, 1, 2, 1},
                            {13, 14, 15, 16, 3}};

   vector<vector<int>> B = {{1, 2, 3, 4, 0},
                            {5, 6, 7, 8, 1},
                            {9, 10, 11, 12, 2},
                            {1, 2, 1, 2, 1},
                            {13, 14, 15, 16, 3}};

   EXPECT_ANY_THROW(strassenMultiply(A, B));
}

TEST(StrassenTest, InvalidDimensions_ABNotEqual)
{
   vector<vector<int>> A = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12},
                            {9, 10, 11, 12},
                            {13, 14, 15, 16}};

   vector<vector<int>> B = {{17, 18, 19, 20},
                            {21, 22, 23, 24},
                            {25, 26, 27, 28},
                            {29, 30, 31, 32}};

   EXPECT_ANY_THROW(strassenMultiply(A, B));
}

TEST(StrassenTest, Multiply)
{
   vector<vector<int>> A = {{1, 2, 3, 4},
                            {5, 6, 7, 8},
                            {9, 10, 11, 12},
                            {13, 14, 15, 16}};

   vector<vector<int>> B = {{17, 18, 19, 20},
                            {21, 22, 23, 24},
                            {25, 26, 27, 28},
                            {29, 30, 31, 32}};

   vector<vector<int>> C_expected = {{250, 260, 270, 280},
                                     {618, 644, 670, 696},
                                     {986, 1028, 1070, 1112},
                                     {1354, 1412, 1470, 1528}};

   auto C = strassenMultiply(A, B);
   printMatrix(C);

   for(auto i=0; i<A.size(); ++i)
   {
      for(auto j=0; j<A[0].size(); ++j)
      {
         ASSERT_EQ(C_expected[i][j], C[i][j]);
      }
   }
}

