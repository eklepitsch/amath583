
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace std;

void hello_world()
{
   cout << "Hello world" << endl;
}

double f_x(double x)
{
   return sqrtl(1 + std::pow((1/x) - 0.25*x, 2));
}

double x_squared(double x)
{
   return x * x;
}

double riemann(std::function<double(double)> f, double xi, double dx,
               unsigned n)
{
   double sum = 0;
   for(auto i=0; i<n; ++i)
   {
      double x = xi + dx * i;
      double area = dx * f(x);
      sum += area;
   }
   return sum;
}

double riemann(std::function<double(double)> f, double xi,
               double xf, unsigned npoints, int rank, int nthreads)
{
   unsigned points_per_thread = npoints / nthreads;
   unsigned leftover_points = npoints % nthreads;
   double dx = (xf - xi)/npoints;

   double local_xi = xi + rank * dx * points_per_thread;
   double local_npoints = points_per_thread;
   // The last thread gets the leftovers
   if(rank == nthreads - 1){local_npoints += leftover_points;}
   
   double local_sum = riemann(f, local_xi, dx, local_npoints);
   double global_sum;
   MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM,
              0, MPI_COMM_WORLD);
   return global_sum;
}

static const double EXACT_SOLUTION_PROBLEM_6 = \
   ((double)35/(double)8) + std::log((double)6);

double riemann_sum_mpi(std::function<double(double)> f, double xi, double xf,
                       double npoints)
{
   double retVal = 0;
   MPI_Init(nullptr, nullptr);

   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   double sum = riemann(f_x, 1, 6, npoints, rank, size);
   //double sum = riemann(x_squared, 0, 1, 1E6, rank, size);

   if(rank == 0)
   {
      cout << "Rieman sum (" << size << " threads): " <<
         std::setprecision(10) << sum << endl;

      double error = std::fabs(sum - EXACT_SOLUTION_PROBLEM_6);
      std::cout << std::setprecision(0) << "log(error) (" << npoints << " points): "
         << std::setprecision(10) << std::log10(error) << std::endl;

      retVal = sum;
   }

   MPI_Finalize();
   return retVal;
}

int main(int argc, char **argv)
{
   if(!std::strcmp(argv[1], "a"))
   {
      auto sum = riemann_sum_mpi(f_x, 1, 6, 1E8);
   }
   else if(!std::strcmp(argv[1], "b"))
   {
      int npoints = std::stoi(std::string(argv[2]));
      if(npoints > 0 && npoints <= 1.e9)
      {
         auto sum = riemann_sum_mpi(f_x, 1, 6, npoints);
      }
      else
      {
         std::cout << "Invalid number of partition points" << std::endl;
      }
   }
   else
   {
      std::cout << "Invalid option. Use 'a' for 7a and 'b' for 7b." << std::endl;
   }

   return 0;
}