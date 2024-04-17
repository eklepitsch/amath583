#include <algorithm>
#include <iostream>
#include <vector>

void swapRows(std::vector<double> &matrix, int nRows, int nCols, int i, int j)
{
   // Assuming nRows,nCols > 0, and i,j are zero-based indices

   // Bounds check
   if(nRows < 1 || nCols < 1 || i < 0 || j < 0 || i >= nRows || j >= nCols ||
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

   for(auto m=0; m<nRows; ++m)
   {
      auto k1 = i*nRows+m;  // Column-major index of the swap value in row i
      auto k2 = j*nRows+m;  // Column major index of the swap value in row j
      auto temp = matrix[k1];
      matrix[k1] = matrix[k2];
      matrix[k2] = temp;
   }
}
