
#include<algorithm>
#include <chrono>
#include <cmath>
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <thread>
#include<utility>
#include<vector>

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

double riemann_sum_mpi(std::function<double(double)> f, double xi,
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

int main()
{
   //hello_world();
   MPI_Init(nullptr, nullptr);
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   double sum = riemann_sum_mpi(f_x, 1, 6, 1E9, rank, size);
   //double sum = riemann_sum_mpi(x_squared, 0, 1, 1E6, rank, size);

   if(rank == 0)
   {
      cout << "Rieman sum: " << std::setprecision(10) << sum << endl;
   }

   MPI_Finalize();
   return 0;
}