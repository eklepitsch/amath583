#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include <iostream>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

std::mutex cout_mtx;

void sequentialTranspose(std::vector<int> &matrix, int rows, int cols)
{
   std::vector<int> copy(rows * cols);
   for (int i = 0; i < rows; ++i)
   {
      for (int j = 0; j < cols; ++j)
      {
         copy[i * cols + j] = matrix[j * rows + i];
      }
   }
   matrix = copy;
}

void kernel(std::vector<int> &matrix, std::vector<int> &copy, int rows, int cols,
            int startingRow, int endingRow, int rank)
{
   if(startingRow < 0 || startingRow >= rows || endingRow < 0 ||
      endingRow > rows || startingRow > endingRow)
   {
      cout_mtx.lock();
      std::cout << "Invalid bounds for thread " << rank << std::endl;
      cout_mtx.unlock();
      return;
   }

   for(int i = startingRow; i < endingRow; ++i)
   {
      for(int j = 0; j < cols; ++j)
      {
         copy[i * cols + j] = matrix[j * rows + i];
      }
   }
}

void threadedTranspose(std::vector<int> &matrix, int rows, int cols, int nthreads)
{
   std::vector<std::thread> threads;
   std::vector<int> copy(rows * cols);

   int leftovers = rows % nthreads;
   for (int i = 0; i < nthreads; ++i)
   {
      int starting_row = i * rows / nthreads;
      int ending_row = (i + 1) * rows / nthreads;
      if(i == nthreads - 1)
      {
         // Last thread gets the leftover rows that did not divide evenly.
         ending_row += leftovers;
      }
      threads.push_back(std::thread(kernel, std::ref(matrix), std::ref(copy),
                                    rows, cols, starting_row, ending_row, i));
   }

   for (auto &t : threads)
   {
      t.join();
   }

   matrix = copy;
}

#endif // TRANSPOSE_HPP