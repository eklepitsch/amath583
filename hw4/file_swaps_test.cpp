#include "file_swaps.hpp"
#include "matrix-utils.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

TEST(FileSwapsTest, Simple)
{
   size_t numRows = 3;
   size_t numCols = 3;
   std::string filename = "./artifacts/matrix-swap-file";

   auto matrix = GenerateRandomMatrix<double>(numRows, numCols);
   std::fstream file(filename, std::ios::out | std::ios::binary);
   file.write(reinterpret_cast<char*>(matrix->data()),
              matrix->size() * sizeof(double));
   file.close();

   std::pair<int, int> indices = GetRandomIndices(3);
   auto i = indices.first;
   auto j = indices.second;

   std::fstream fileToSwap(filename, std::ios::in | std::ios::out |
                           std::ios::binary);
   //auto startTime = std::chrono::high_resolution_clock::now();
   swapRowsInFile(fileToSwap, numRows, numCols, i, j);
   //auto endTime = std::chrono::high_resolution_clock::now();
   //std::chrono::duration<double> duration = endTime - startTime;
   // Close the file after swapping
   fileToSwap.close();
   // ...
   // after each problem size delete the test file
   //std::remove(filename.c_str());
}