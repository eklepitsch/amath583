#include<algorithm>
#include <chrono>
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include <fstream>
#include <iostream>
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

int main()
{
   //problem_4();
   problem_5();
   return 0;
}