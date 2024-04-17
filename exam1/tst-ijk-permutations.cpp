#include "exam1.problem1.hpp"
#undef NDEBUG
#include <cassert>
#include <iostream>

constexpr int m = 2;
constexpr int n = 4;
constexpr int p = 3;

// A = m x p
// [ 1, 2, 3 ]
// [ 4, 5, 6 ]
static const std::vector<int> A = {1, 4, 2, 5, 3, 6};

// B = p x n
// [ 1, 2, 3, 4 ]
// [ 3, 4, 7, 8 ]
// [ 9, 0, 1, 2 ]
static const std::vector<int> B = {1, 3, 9, 2, 4, 0, 3, 7, 1, 4, 8, 2};

// C = m x n
// [ 1, 2, 3, 4 ]
// [ 5, 6, 7, 8 ]
static const std::vector<int> C = {1, 5, 2, 6, 3, 7, 4, 8};

// Expected result for alpha=1, beta=1
// [ 35, 12, 23, 30 ]
// [ 78, 34, 60, 76 ]
static const std::vector<int> R1 = {35, 78, 12, 34, 23, 60, 30, 76};

// Expected result for alpha=2, beta=3
// [  71, 26,  49,  64 ]
// [ 161, 74, 127, 160 ]
static const std::vector<int> R2 = {71, 161, 26, 74, 49, 127, 64, 160};

void print_matrix(const std::vector<int>& matrix, int m, int n)
{
   for(auto i=0; i<m; ++i)
   {
      std::cout << "[ ";
      for(auto j=0; j<n; ++j)
      {
         auto k = j*m+i;
         std::cout << matrix[k] << " ";
      }
      std::cout << "]" << std::endl;
   }
}

void check_result(std::vector<int>& result, const std::vector<int>& expected)
{
   assert(result.size() == expected.size());
   for(auto i=0; i<expected.size(); ++i)
   {
      assert(result[i] == expected[i]);
   }
}

void test_case_invalid_A_dims()
{
   std::cout << "Test case invalid A dims" << std::endl;
   const std::vector<int> iA(0, m*p+1);
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_ijk(1, iA, B, 1, mC, m, p, n);
}

void test_case_invalid_B_dims()
{
   std::cout << "Test case invalid B dims" << std::endl;
   const std::vector<int> iB(0, p*n+1);
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_ijk(1, A, iB, 1, mC, m, p, n);
}

void test_case_invalid_C_dims()
{
   std::cout << "Test case invalid C dims" << std::endl;
   std::vector<int> iC(0, m*n+1);
   mm_ijk(1, A, B, 1, iC, m, p, n);
}

void test_case_ijk_alpha_beta_equal_1()
{
   std::cout << "Test case IJK alpha=1, beta=1" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_ijk(1, A, B, 1, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1);
}

void test_case_ijk_alpha_2_beta_3()
{
   std::cout << "Test case IJK alpha=2, beta=3" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_ijk(2, A, B, 3, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2);
}

void test_case_ikj_alpha_beta_equal_1()
{
   std::cout << "Test case IKJ alpha=1, beta=1" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_ikj(1, A, B, 1, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1);
}

void test_case_ikj_alpha_2_beta_3()
{
   std::cout << "Test case IKJ alpha=2, beta=3" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_ikj(2, A, B, 3, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2);
}

void test_case_jik_alpha_beta_equal_1()
{
   std::cout << "Test case JIK alpha=1, beta=1" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_jik(1, A, B, 1, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1);
}

void test_case_jik_alpha_2_beta_3()
{
   std::cout << "Test case JIK alpha=2, beta=3" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_jik(2, A, B, 3, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2);
}

void test_case_jki_alpha_beta_equal_1()
{
   std::cout << "Test case JKI alpha=1, beta=1" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_jki(1, A, B, 1, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1);
}

void test_case_jki_alpha_2_beta_3()
{
   std::cout << "Test case JKI alpha=2, beta=3" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_jki(2, A, B, 3, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2);
}

void test_case_kij_alpha_beta_equal_1()
{
   std::cout << "Test case KIJ alpha=1, beta=1" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_kij(1, A, B, 1, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1);
}

void test_case_kij_alpha_2_beta_3()
{
   std::cout << "Test case KIJ alpha=2, beta=3" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_kij(2, A, B, 3, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2);
}

void test_case_kji_alpha_beta_equal_1()
{
   std::cout << "Test case KJI alpha=1, beta=1" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_kji(1, A, B, 1, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1);
}

void test_case_kji_alpha_2_beta_3()
{
   std::cout << "Test case KJI alpha=2, beta=3" << std::endl;
   std::vector<int> mC;
   for(auto i : C){mC.push_back(i);}
   mm_kji(2, A, B, 3, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2);
}

int main()
{
   std::cout << "The matrix A:" << std::endl;
   print_matrix(A, m, p);
   std::cout << "The matrix B" << std::endl;
   print_matrix(B, p, n);
   std::cout << "The matrix C" << std::endl;
   print_matrix(C, m, n);

   test_case_invalid_A_dims();
   test_case_invalid_B_dims();
   test_case_invalid_C_dims();
   test_case_ijk_alpha_beta_equal_1();
   test_case_ijk_alpha_2_beta_3();
   test_case_ikj_alpha_beta_equal_1();
   test_case_ikj_alpha_2_beta_3();
   test_case_jik_alpha_beta_equal_1();
   test_case_jik_alpha_2_beta_3();
   test_case_jki_alpha_beta_equal_1();
   test_case_jki_alpha_2_beta_3();
   test_case_kij_alpha_beta_equal_1();
   test_case_kij_alpha_2_beta_3();
   test_case_kji_alpha_beta_equal_1();
   test_case_kji_alpha_2_beta_3();

   std::cout << "PASS" << std::endl;  // Pass if no assertions
}