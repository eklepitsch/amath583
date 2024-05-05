#include "matrix_class.hpp"
#include <gtest/gtest.h>

//class MatrixTest : public test::Test
//{
//  protected:
//
//  Matrix<float> mEmpty2x2_;
//  Matrix<float> mA2x2_;
//  Matrix<float> mB2x2_;
//  Matrix<float> mAplusB2x2_;
//};

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}

TEST(MatrixTest, Construction)
{
  auto m = Matrix<float>(2, 2);
}

TEST(MatrixTest, Addition)
{
  
}
