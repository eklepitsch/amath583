#include <iomanip>
#include <iostream>
#include <limits>
#include "hw2.h"

using namespace std;

void problem_1()
{
    constexpr auto fp_max_precision{numeric_limits<float>::digits10 + 1};
    constexpr auto dp_max_precision{numeric_limits<double>::digits10 + 1};
    constexpr auto ldp_max_precision{numeric_limits<long double>::digits10 + 1};
    const auto default_precision{cout.precision()};

    float sp = measure_precision<float>(powf);
    cout << setprecision(fp_max_precision);
    cout << "Machine single precision: " << sp << endl;

    double dp = measure_precision<double>(pow);
    cout << setprecision(dp_max_precision);
    cout << "Machine double precision: " << dp << endl;

    long double ldp = measure_precision<long double>(powl);
    cout << setprecision(ldp_max_precision);
    cout << "Machine long double precision: " << ldp << endl;

    cout << setprecision(default_precision);
}

void problem_3()
{
    int int_result = multiply_as_type<int>();
    cout << "[int] 200*300*400*500 = " << int_result << endl;

    int long_result = multiply_as_type<long>();
    cout << "[long] 200*300*400*500 = " << long_result << endl;

    int long_long_result = multiply_as_type<long long>();
    cout << "[long long] 200*300*400*500 = " << long_long_result << endl;
}

int main()
{
    problem_1();
    problem_3();

    return 0;
}
