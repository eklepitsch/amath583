#include <algorithm>
#include <iostream>
#include <vector>

void swapRows(std::vector<double> &matrix, int nRows, int nCols, int i, int j)
{
   // Assuming nRows,nCols > 0, and i,j are zero-based indices

   // Bounds check
   if(nRows < 1 || nCols < 1 || i < 0 || j < 0 || i > nRows || j > nCols ||
      matrix.size() != nRows*nCols)
   {
      std::cout << "Invalid dimensions" << std::endl;
      return;
   }
   
   for(auto n=0; n<nCols; ++n)
   {
      auto k1 = n*nRows+i;  // Column-major index of the swap value in row i
      auto k2 = n*nRows+j;  // Column major index of the swap value in row j
      auto temp = matrix[k1];
      matrix[k1] = matrix[k2];
      matrix[k2] = temp;
   }
}

/*Assuming nRows,nCols > 0, and i,j are zero-based indices.*/
void swapCols(std::vector<double> &matrix, int nRows, int nCols, int i, int j)
{
   // Assuming nRows,nCols > 0, and i,j are zero-based indices

   // Bounds check
   if(nRows < 1 || nCols < 1 || i < 0 || j < 0 || i >= nRows || j >= nCols ||
      matrix.size() != nRows*nCols)
   {
      std::cout << "Invalid dimensions" << std::endl;
      return;
   }

   // auto k1_begin = matrix.begin() + nRows*i;
   // auto k1_end = k1_begin + nRows;
   // auto k2_begin = matrix.begin() + nRows*j;
   // std::swap_ranges(k1_begin, k1_end, k2_begin);

   for(auto m=0; m<nRows; ++m)
   {
      auto k1 = i*nCols+m;  // Column-major index of the swap value in row i
      auto k2 = j*nCols+m;  // Column major index of the swap value in row j
      auto temp = matrix[k1];
      matrix[k1] = matrix[k2];
      matrix[k2] = temp;
   }
}
