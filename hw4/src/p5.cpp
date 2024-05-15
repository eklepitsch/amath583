
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
   mkdir("./artifacts", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
   problem_5();
   return 0;
}