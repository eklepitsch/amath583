#include "file_swaps.hpp"
#include "matrix-utils.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

std::vector<double> ReadMatrixFromFile(std::string filename, int nRows, int nCols)
{
   std::fstream file(filename, std::ios::in | std::ios::binary);
   file.seekg(0, std::ios_base::end);
   long length = file.tellg();
   auto fileSize = nRows * nCols * sizeof(double);
   if(!file.good() || length != fileSize)
   {
      throw std::invalid_argument("Invalid file");
   }

   std::vector<double> matrix(nRows * nCols);
   file.seekg(0, std::ios_base::beg);
   file.read(reinterpret_cast<char*>(matrix.data()), fileSize);
   file.close();
   return matrix;
}

TEST(FileSwapsTest, SwapRows)
{
   size_t numRows = 3;
   size_t numCols = 3;
   std::string filename = "./artifacts/matrix-swap-file";

   // Write a matrix to a file
   auto matrix = GenerateRandomMatrix<double>(numRows, numCols);
   std::fstream file(filename, std::ios::out | std::ios::binary);
   file.write(reinterpret_cast<char*>(matrix->data()),
              matrix->size() * sizeof(double));
   file.close();

   // Read back the matrix
   auto matrix_before_swap = ReadMatrixFromFile(filename, 3, 3);
   cout << "Matrix before swap" << endl;
   print_matrix(matrix_before_swap, 3, 3);

   // Swap random rows
   std::pair<int, int> indices = GetRandomIndices(3);
   auto i = indices.first;
   auto j = indices.second;
   std::fstream fileToSwap(filename, std::ios::in | std::ios::out |
                           std::ios::binary);
   cout << "Swapping rows " << i << " and " << j << endl;
   swapRowsInFile(fileToSwap, numRows, numCols, i, j);
   fileToSwap.close();

   // Read back the matrix and check the swap was correct
   auto matrix_after_swap = ReadMatrixFromFile(filename, 3, 3);
   cout << "Matrix after swap" << endl;
   print_matrix(matrix_after_swap, 3, 3);
}

TEST(FileSwapsTest, SwapCols)
{
   size_t numRows = 3;
   size_t numCols = 3;
   std::string filename = "./artifacts/matrix-swap-file";

   // Write a matrix to a file
   auto matrix = GenerateRandomMatrix<double>(numRows, numCols);
   std::fstream file(filename, std::ios::out | std::ios::binary);
   file.write(reinterpret_cast<char*>(matrix->data()),
              matrix->size() * sizeof(double));
   file.close();

   // Read back the matrix
   auto matrix_before_swap = ReadMatrixFromFile(filename, 3, 3);
   cout << "Matrix before swap" << endl;
   print_matrix(matrix_before_swap, 3, 3);

   // Swap random rows
   std::pair<int, int> indices = GetRandomIndices(3);
   auto i = indices.first;
   auto j = indices.second;
   std::fstream fileToSwap(filename, std::ios::in | std::ios::out |
                           std::ios::binary);
   cout << "Swapping cols " << i << " and " << j << endl;
   swapColsInFile(fileToSwap, numRows, numCols, i, j);
   fileToSwap.close();

   // Read back the matrix and check the swap was correct
   auto matrix_after_swap = ReadMatrixFromFile(filename, 3, 3);
   cout << "Matrix after swap" << endl;
   print_matrix(matrix_after_swap, 3, 3);
}