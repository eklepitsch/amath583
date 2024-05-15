
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
#include<sys/stat.h>
#include <thread>
#include<utility>
#include<vector>
#include"file_swaps.hpp"
#include "matrix-utils.hpp"

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
      if(i == nthreads - 1)
      {
         // The last thread gets the leftovers
         threads.emplace_back(riemann, f, xi + i * dx * points_per_thread, dx,
                              points_per_thread + leftover_points,
                              std::ref(sum_mutex), std::ref(sum));
      }
      else
      {
         threads.emplace_back(riemann, f, xi + i * dx * points_per_thread, dx,
                              points_per_thread, std::ref(sum_mutex), std::ref(sum));
      }
   }
   for(auto i = 0; i<nthreads; ++i)
   {
      threads[i].join();
   }
   return sum;
}

static const double EXACT_SOLUTION_PROBLEM_6 = \
   ((double)35/(double)8) + std::log((double)6);

void problem_6()
{
   std::ofstream scaling_results;
   scaling_results.open("./artifacts/scaling-results.csv");
   std::ofstream error_results;
   error_results.open("./artifacts/error-results.csv");

   unsigned n = 1;
   while(n <= 16)
   {
      auto start= std::chrono::high_resolution_clock::now();
      auto sum = riemann_sum_cxx_threads(f_x, 1, 6, 1E8, n);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

      scaling_results << std::fixed << std::setprecision(0) << n << ", " <<
         std::scientific << std::setprecision(10) << 
         std::fabs(sum - EXACT_SOLUTION_PROBLEM_6) << ", " <<
         sum << ", " << duration.count()*1.e-9 << std::endl;
      
      n <<= 1;
   }

   n = 10;
   while(n <= 1E6)
   {
      auto start= std::chrono::high_resolution_clock::now();
      auto sum = riemann_sum_cxx_threads(f_x, 1, 6, n, 8);
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

      double error = std::fabs(sum - EXACT_SOLUTION_PROBLEM_6);
      error_results << std::fixed << std::setprecision(0) << n << ", " <<
         std::scientific << std::setprecision(10) << error << ", " <<
         std::log10(error) << std::endl;

      n *= 10;
   }

   // Test code
   auto sum = riemann_sum_cxx_threads(f_x, 1, 6, 1000000, 9);
   //auto sum = riemann_sum_cxx_threads(x_squared, 0, 1, 1000000, 10);
   std::cout << "Rieman sum: " << std::setprecision(10) << sum << std::endl;

   scaling_results.close();
   error_results.close();
}

int main()
{
   mkdir("./artifacts", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   problem_6();
   return 0;
}