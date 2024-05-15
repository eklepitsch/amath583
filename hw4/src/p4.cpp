
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

int main()
{
   mkdir("./artifacts", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   problem_4();
   return 0;
}