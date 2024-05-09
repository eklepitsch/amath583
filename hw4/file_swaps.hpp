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

   for(auto m=0; m<nRows; ++m)
   {
      auto k1 = i*nRows+m;  // Column-major index of the swap value in row i
      auto k2 = j*nRows+m;  // Column major index of the swap value in row j

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

#endif // FILE_SWAPS_HPP