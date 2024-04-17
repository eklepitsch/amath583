#include "exam1.problem1.hpp"
#undef NDEBUG
#include <cassert>
#include <iomanip>
#include <iostream>

constexpr int m = 2;
constexpr int n = 4;
constexpr int p = 3;

// A = m x p
// [ 1, 2, 3 ]
// [ 4, 5, 6 ]
template<typename T>
static const std::vector<T> A = {1, 4, 2, 5, 3, 6};

// B = p x n
// [ 1, 2, 3, 4 ]
// [ 3, 4, 7, 8 ]
// [ 9, 0, 1, 2 ]
template<typename T>
static const std::vector<T> B = {1, 3, 9, 2, 4, 0, 3, 7, 1, 4, 8, 2};

// C = m x n
// [ 1, 2, 3, 4 ]
// [ 5, 6, 7, 8 ]
template<typename T>
static const std::vector<T> C = {1, 5, 2, 6, 3, 7, 4, 8};

// Expected result for alpha=1, beta=1
// [ 35, 12, 23, 30 ]
// [ 78, 34, 60, 76 ]
template<typename T>
static const std::vector<T> R1 = {35, 78, 12, 34, 23, 60, 30, 76};

// Expected result for alpha=2, beta=3
// [  71, 26,  49,  64 ]
// [ 161, 74, 127, 160 ]
template<typename T>
static const std::vector<T> R2 = {71, 161, 26, 74, 49, 127, 64, 160};

template<typename T>
void print_matrix(const std::vector<T>& matrix, int m, int n)
{
   for(auto i=0; i<m; ++i)
   {
      std::cout << "[ ";
      for(auto j=0; j<n; ++j)
      {
         auto k = j*m+i;
         std::cout << std::fixed << std::setprecision(2) << matrix[k] << " ";
      }
      std::cout << "]" << std::endl;
   }
}

template<typename T>
void check_result(std::vector<T>& result, const std::vector<T>& expected)
{
   assert(result.size() == expected.size());
   for(auto i=0; i<expected.size(); ++i)
   {
      assert(result[i] == expected[i]);
   }
}

template<typename T>
void test_case_invalid_A_dims()
{
   std::cout << "Test case invalid A dims" << std::endl;
   const std::vector<T> iA(0, m*p+1);
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 1;
   T b = 1;
   mm_ijk(a, iA, B<T>, b, mC, m, p, n);
}

template<typename T>
void test_case_invalid_B_dims()
{
   std::cout << "Test case invalid B dims" << std::endl;
   const std::vector<T> iB(0, p*n+1);
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 1;
   T b = 1;
   mm_ijk(a, A<T>, iB, b, mC, m, p, n);
}

template<typename T>
void test_case_invalid_C_dims()
{
   std::cout << "Test case invalid C dims" << std::endl;
   std::vector<T> iC(0, m*n+1);
   T a = 1;
   T b = 1;
   mm_ijk(a, A<T>, B<T>, b, iC, m, p, n);
}

template<typename T>
void test_case_ijk_alpha_beta_equal_1()
{
   std::cout << "Test case IJK alpha=1, beta=1" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 1;
   T b = 1;
   mm_ijk(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1<T>);
}

template<typename T>
void test_case_ijk_alpha_2_beta_3()
{
   std::cout << "Test case IJK alpha=2, beta=3" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 2;
   T b = 3;
   mm_ijk(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2<T>);
}

template<typename T>
void test_case_ikj_alpha_beta_equal_1()
{
   std::cout << "Test case IKJ alpha=1, beta=1" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 1;
   T b = 1;
   mm_ikj(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1<T>);
}

template<typename T>
void test_case_ikj_alpha_2_beta_3()
{
   std::cout << "Test case IKJ alpha=2, beta=3" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 2;
   T b = 3;
   mm_ikj(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2<T>);
}

template<typename T>
void test_case_jik_alpha_beta_equal_1()
{
   std::cout << "Test case JIK alpha=1, beta=1" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 1;
   T b = 1;
   mm_jik(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1<T>);
}

template<typename T>
void test_case_jik_alpha_2_beta_3()
{
   std::cout << "Test case JIK alpha=2, beta=3" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 2;
   T b = 3;
   mm_jik(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2<T>);
}

template<typename T>
void test_case_jki_alpha_beta_equal_1()
{
   std::cout << "Test case JKI alpha=1, beta=1" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 1;
   T b = 1;
   mm_jki(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1<T>);
}

template<typename T>
void test_case_jki_alpha_2_beta_3()
{
   std::cout << "Test case JKI alpha=2, beta=3" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 2;
   T b = 3;
   mm_jki(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2<T>);
}

template<typename T>
void test_case_kij_alpha_beta_equal_1()
{
   std::cout << "Test case KIJ alpha=1, beta=1" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 1;
   T b = 1;
   mm_kij(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1<T>);
}

template<typename T>
void test_case_kij_alpha_2_beta_3()
{
   std::cout << "Test case KIJ alpha=2, beta=3" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 2;
   T b = 3;
   mm_kij(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2<T>);
}

template<typename T>
void test_case_kji_alpha_beta_equal_1()
{
   std::cout << "Test case KJI alpha=1, beta=1" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 1;
   T b = 1;
   mm_kji(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "AB + C: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R1<T>);
}

template<typename T>
void test_case_kji_alpha_2_beta_3()
{
   std::cout << "Test case KJI alpha=2, beta=3" << std::endl;
   std::vector<T> mC;
   for(auto i : C<T>){mC.push_back(i);}
   T a = 2;
   T b = 3;
   mm_kji(a, A<T>, B<T>, b, mC, m, p, n);
   std::cout << "aAB + bC: " << std::endl;
   print_matrix(mC, m, n);
   check_result(mC, R2<T>);
}

template<typename T>
void run_tests()
{
   test_case_invalid_A_dims<T>();
   test_case_invalid_B_dims<T>();
   test_case_invalid_C_dims<T>();
   test_case_ijk_alpha_beta_equal_1<T>();
   test_case_ijk_alpha_2_beta_3<T>();
   test_case_ikj_alpha_beta_equal_1<T>();
   test_case_ikj_alpha_2_beta_3<T>();
   test_case_jik_alpha_beta_equal_1<T>();
   test_case_jik_alpha_2_beta_3<T>();
   test_case_jki_alpha_beta_equal_1<T>();
   test_case_jki_alpha_2_beta_3<T>();
   test_case_kij_alpha_beta_equal_1<T>();
   test_case_kij_alpha_2_beta_3<T>();
   test_case_kji_alpha_beta_equal_1<T>();
   test_case_kji_alpha_2_beta_3<T>();
}

int main()
{
   std::cout << "The matrix A:" << std::endl;
   print_matrix<int>(A<int>, m, p);
   std::cout << "The matrix B" << std::endl;
   print_matrix<int>(B<int>, p, n);
   std::cout << "The matrix C" << std::endl;
   print_matrix<int>(C<int>, m, n);

   std::cout << "RUNNING TESTS FOR TYPE: int" << std::endl;
   run_tests<int>();
   std::cout << "RUNNING TESTS FOR TYPE: float" << std::endl;
   run_tests<float>();
   std::cout << "RUNNING TESTS FOR TYPE: double" << std::endl;
   run_tests<double>();

   std::cout << "PASS" << std::endl;  // Pass if no assertions
}