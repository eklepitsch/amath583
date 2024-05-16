#ifndef FILE_SWAPS_HPP
#define FILE_SWAPS_HPP

#include <fstream>
#include <stdexcept>
#include <iostream>

void swapRowsInFile(std::fstream& file, int nRows, int nCols, int i, int j)
{
   // Assuming nRows,nCols > 0, and i,j are zero-based indices

   // Bounds check
   if(nRows < 1 || nCols < 1 || i < 0 || j < 0 || i >= nRows || j >= nRows)
   {
      throw std::invalid_argument("Invalid matrix dimensions");
   }

   // File check
   file.seekg(0, std::ios_base::end);
   auto length = file.tellg();
   if(!file.good() || length != nRows*nCols*sizeof(double))
   {
      throw std::invalid_argument("Invalid file");
   }

   for(auto n=0; n<nCols; ++n)
   {
      auto k1 = n*nRows+i;  // Column-major index of the swap value in row i
      auto k2 = n*nRows+j;  // Column major index of the swap value in row j

      double temp1;
      file.seekg(k1 * sizeof(double), std::ios_base::beg);
      file.read(reinterpret_cast<char*>(&temp1), sizeof(double));

      double temp2;
      file.seekg(k2 * sizeof(double), std::ios_base::beg);
      file.read(reinterpret_cast<char*>(&temp2), sizeof(double));

      file.seekp(k1 * sizeof(double), std::ios_base::beg);
      file.write(reinterpret_cast<char*>(&temp2), sizeof(double));

      file.seekp(k2 * sizeof(double), std::ios_base::beg);
      file.write(reinterpret_cast<char*>(&temp1), sizeof(double));
   }
}

void swapColsInFile(std::fstream& file, int nRows, int nCols, int i, int j)
{
   // Assuming nRows,nCols > 0, and i,j are zero-based indices

   // Bounds check
   if(nRows < 1 || nCols < 1 || i < 0 || j < 0 || i >= nCols || j >= nCols)
   {
      throw std::invalid_argument("Invalid matrix dimensions");
   }

   // File check
   file.seekg(0, std::ios_base::end);
   auto length = file.tellg();
   if(!file.good() || length != nRows*nCols*sizeof(double))
   {
      throw std::invalid_argument("Invalid file");
   }

   auto k1_start = i*nRows;  // Column-major index of the first element in the first column
   auto k2_start = j*nRows;  // Column-major index of the first element in the second column

   double temp1[nRows];  // Temporary storage area for a column
   double temp2[nRows];

   // Since we are in column major order, we can swap entire columns at a time.
   file.seekg(k1_start * sizeof(double), std::ios_base::beg);
   file.read(reinterpret_cast<char*>(temp1), sizeof(double)*nRows);
   file.seekg(k2_start * sizeof(double), std::ios_base::beg);
   file.read(reinterpret_cast<char*>(temp2), sizeof(double)*nRows);
   
   file.seekp(k1_start * sizeof(double), std::ios_base::beg);
   file.write(reinterpret_cast<char*>(temp2), sizeof(double)*nRows);
   file.seekp(k2_start * sizeof(double), std::ios_base::beg);
   file.write(reinterpret_cast<char*>(temp1), sizeof(double)*nRows);
}

#endif // FILE_SWAPS_HPP