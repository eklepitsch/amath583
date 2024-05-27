#include<iostream>
#include<complex>
#include<cstdlib>
#include<string>
#include<cmath>
#include<vector>
#include<chrono>
#include<limits>
#include<fstream>
#include<iomanip>
#include<cblas.h>
#include<lapacke.h>

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <max_problem_size>" << std::endl;
        return 1;
    }
    int max_problem_size = std::stoi(argv[1]);

    std::ofstream results("artifacts/p1-results.csv");
    results << "problem_size, residual, log(residual),"
               " normalized error, log(normalized error)" << std::endl; 
    int problem_size = 16;

    while(problem_size <= max_problem_size)
    {
        int ma = problem_size;
        int na = problem_size;

        auto a=(std::complex<double>*)malloc(sizeof(std::complex<double>) * ma * na);
        auto a_orig=(std::complex<double>*)malloc(sizeof(std::complex<double>) * ma * na);
        auto b=(std::complex<double>*)malloc(sizeof(std::complex<double>) * ma);
        auto b_orig=(std::complex<double>*)malloc(sizeof(std::complex<double>) * ma);
        auto b_resid=(std::complex<double>*)malloc(sizeof(std::complex<double>) * ma);
        auto z=(std::complex<double>*)malloc(sizeof(std::complex<double>) * na);

        srand(0);
        int k=0;
        for(int j=0; j<na; j++) {
            for(int i=0; i<ma; i++) {
                a[k] = 0.5 - (double)rand()/(double)RAND_MAX
                        + std::complex<double>(0,1) * (0.5-(double)rand()/(double)RAND_MAX);
                if(i==j) a[k]*=static_cast<double>(ma);
                a_orig[k] = a[k];
                k++;
            }
        }
        srand(1);
        for(int i=0; i<ma; i++) {
            b[i] = 0.5 - (double)rand()/(double)RAND_MAX
                    + std::complex<double>(0,1) * (0.5-(double)rand()/(double)RAND_MAX);
            b_orig[i] = b[i];
            b_resid[i] = b[i];
        }

        // Solve the system of linear equations (store result in b)
        lapack_int* ipiv = (lapack_int*)malloc(sizeof(lapack_int)*ma);
        LAPACKE_zgesv(LAPACK_COL_MAJOR, // int matrix_order
                      ma,               // lapack_int n
                      1,                // lapack_int nrhs
                      reinterpret_cast<lapack_complex_double*>(a), // lapack_complex_double∗ a
                      ma,               // lapack_int lda
                      ipiv,             // lapack_int∗ ipiv
                      reinterpret_cast<lapack_complex_double*>(b), // lapack_complex_double∗ b
                      ma);              // lapack_int ldb

        // Copy result to z
        for(int i=0; i<ma; i++) z[i] = b[i];
                    
        // Calculate r = b - Az (store in b_resid)
        auto alpha = std::complex<double>(-1.0, 0.0);
        auto beta = std::complex<double>(1.0, 0.0);
        cblas_zgemv(CblasColMajor, CblasNoTrans, ma, na, &alpha,
                    a_orig, ma, z, 1, &beta, b_resid, 1);

        // Residual is the 2-norm of (b - Az)
        double residual = cblas_dznrm2(ma, b_resid, 1);

        // Calculate the infinity norm of A (max absolute row sum)
        double a_infty_norm = 0.0;
        for(int i=0; i<ma; ++i)
        {
            double row_sum = 0.0;
            for(int j=0; j<na; ++j)
            {
                auto k = j * ma + i; // Column major index
                row_sum += std::abs(a_orig[k]);
            }
            a_infty_norm = std::max(a_infty_norm, row_sum);
        }

        // Get machine epsilon
        double machine_epsilon = std::numeric_limits<double>::epsilon();
        
        // Calculate the 2-norm of z
        double z_norm = cblas_dznrm2(na, z, 1);

        // Calculate the normalized error
        double normalized_error = residual / (a_infty_norm * z_norm * machine_epsilon);
        std::cout << residual << " / ( " << a_infty_norm << " * " << z_norm << " * "
            << machine_epsilon << " ) = " << normalized_error << std::endl;

        free(a);
        free(b);
        free(z);
        free(a_orig);
        free(b_orig);
        free(b_resid);
        free(ipiv);
        results << problem_size << ", " << std::setprecision(10) << residual << ", "
            << std::log10(residual) << ", " << normalized_error << ", "
            << std::log10(normalized_error) << std::endl;

        problem_size <<= 1;
    }
}
