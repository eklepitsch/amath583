#include<algorithm>
#include <chrono>
#include <cmath>
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include<utility>
#include<vector>
#include"file_swaps.hpp"
#include "matrix-utils.hpp"


long double write_square_matrix_to_file(size_t n)
{
   auto m = GenerateSquareMatrix<double>(n);
   std::ofstream myFile("./artifacts/square-matrix-" + std::to_string(n),
                        std::ios::binary | std::ios::trunc);
   auto start = std::chrono::high_resolution_clock::now();
   myFile.write(reinterpret_cast<char*>(m->data()),
                m->size() * sizeof(double));
   auto stop = std::chrono::high_resolution_clock::now();
   myFile.close();

   if(n == 4)
   {
      std::cout << "Matrix written, n=4:" << std::endl;
      print_matrix(*m, n, n);
   }

   auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
   return duration.count()*1.e-9;  // Convert duration to seconds
}

void write_square_matrices_to_file()
{
   std::ofstream write_results;
   write_results.open("./artifacts/write-matrix-results.csv");

   unsigned n = 2;
   while(n <= 16384)
   {
      auto duration = write_square_matrix_to_file(n);
      long double bytes = sizeof(double) * n * n;
      write_results << std::fixed << std::setprecision(0) << n << ", " <<
         bytes << ", " <<
         std::scientific << std::setprecision(10) << duration << ", " <<
         bytes / duration << std::endl;
      n <<= 1;
   }
}

long double read_square_matrix_from_file(size_t n)
{
   auto m = GenerateSquareMatrix<double>(n);
   std::ifstream myFile("./artifacts/square-matrix-" + std::to_string(n),
                        std::ios::binary);
   auto start = std::chrono::high_resolution_clock::now();
   long double bytes = sizeof(double) * n * n;
   myFile.read(reinterpret_cast<char*>(m->data()), bytes);
   auto stop = std::chrono::high_resolution_clock::now();
   myFile.close();

   if(n == 4)
   {
      std::cout << "Matrix read, n=4:" << std::endl;
      print_matrix(*m, n, n);
   }

   auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
   return duration.count()*1.e-9;  // Convert duration to seconds
}

void read_square_matrices_from_file()
{
   std::ofstream read_results;
   read_results.open("./artifacts/read-matrix-results.csv");

   unsigned n = 2;
   while(n <= 16384)
   {
      auto duration = read_square_matrix_from_file(n);
      long double bytes = sizeof(double) * n * n;
      read_results << std::fixed << std::setprecision(0) << n << ", " <<
         bytes << ", " <<
         std::scientific << std::setprecision(10) << duration << ", " <<
         bytes / duration << std::endl;
      n <<= 1;
   }
}

void problem_4()
{
   write_square_matrices_to_file();
   read_square_matrices_from_file();
}

long double swap_rows_in_square_matrix(std::fstream& file, int n,
                                       unsigned ntrials)
{
   long double elapsedtime = 0.L;
   long double avgtime;

   for(unsigned k=0; k<ntrials; ++k)
   {
      auto indices = GetRandomIndices(n);
      auto i = indices.first;
      auto j = indices.second;
      auto start= std::chrono::high_resolution_clock::now();
      swapRowsInFile(file, n, n, i, j);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
   }
   avgtime = elapsedtime/static_cast<long double>(ntrials);
   return avgtime;
}

long double swap_cols_in_square_matrix(std::fstream& file, int n,
                                       unsigned ntrials)
{
   long double elapsedtime = 0.L;
   long double avgtime;

   for(unsigned k=0; k<ntrials; ++k)
   {
      auto indices = GetRandomIndices(n);
      auto i = indices.first;
      auto j = indices.second;
      auto start= std::chrono::high_resolution_clock::now();
      swapColsInFile(file, n, n, i, j);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
   }
   avgtime = elapsedtime/static_cast<long double>(ntrials);
   return avgtime;
}

void problem_5()
{
   long double row_duration, column_duration;
   std::string matrix_file = "./artifacts/matrix-swap-file";
   std::ofstream results_file;
   results_file.open("./artifacts/file-swaps-durations.csv");

   int n = 16;
   while(n <= 8192)
   {
      auto matrix = GenerateSquareMatrix<double>(n);
      std::fstream file(matrix_file, std::ios::out | std::ios::binary |
                        std::ios::trunc);
      file.write(reinterpret_cast<char*>(matrix->data()),
                 matrix->size() * sizeof(double));
      file.flush();
      file.close();

      std::fstream fileToSwap(matrix_file, std::ios::in | std::ios::out |
                              std::ios::binary);
      row_duration = swap_rows_in_square_matrix(fileToSwap, n, 5);
      column_duration = swap_cols_in_square_matrix(fileToSwap, n, 5);
      fileToSwap.close();

      results_file << std::fixed << std::setprecision(0) << n << ", " <<
         std::scientific << std::setprecision(10) << row_duration << ", " <<
         column_duration << std::endl;

      n <<= 1;
   }

   results_file.close();
}

double f_x(double x)
{
   return sqrtl(1 + std::pow((1/x) - 0.25*x, 2));
}

double x_squared(double x)
{
   return x * x;
}

void riemann(std::function<double(double)> f, double xi, double dx,
             unsigned n, std::mutex& mtx, double& shared_sum)
{
   double sum = 0;
   for(auto i=0; i<n; ++i)
   {
      double x = xi + dx * i;
      double area = dx * f(x);
      sum += area;
   }

   mtx.lock();
   shared_sum += sum;
   mtx.unlock();
}

double riemann_sum_cxx_threads(std::function<double(double)> f, double xi,
                               double xf, unsigned npoints, unsigned nthreads)
{
   unsigned points_per_thread = npoints / nthreads;
   unsigned leftover_points = npoints % nthreads;
   double dx = (xf - xi)/npoints;

   std::mutex sum_mutex;
   double sum = 0;

   std::vector<std::thread> threads;
   threads.reserve(nthreads);
   for(auto i = 0; i<nthreads; ++i)
   {
      threads.emplace_back(riemann, f, xi + i * dx * points_per_thread, dx, points_per_thread,
                           std::ref(sum_mutex), std::ref(sum));
   }
   for(auto i = 0; i<nthreads; ++i)
   {
      threads[i].join();
   }
   return sum;
}

void problem_6()
{
   auto sum = riemann_sum_cxx_threads(f_x, 1, 6, 1000000, 10);
   //auto sum = riemann_sum_cxx_threads(x_squared, 0, 1, 1000000, 10);
   std::cout << "Rieman sum: " << std::setprecision(10) << sum << std::endl;
}

int main()
{
   //problem_4();
   //problem_5();
   problem_6();
   return 0;
}