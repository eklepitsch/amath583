#include "transpose.hpp"
#include "matrix-utils.hpp"
#include <iostream>

using namespace std;

void copy_matrix(vector<int>& new_matrix, const vector<int>& old_matrix, int size)
{
   for(int i = 0; i < size; ++i)
   {
      new_matrix[i] = old_matrix[i];
   }
}

int main(int argc, char** argv)
{
   if (argc != 4)
   {
       cerr << "Usage: " << argv[0] << " <n> <m> <nthreads>" << endl;
       return 1;
   }

   int m = atoi(argv[1]); // Rows
   int n = atoi(argv[2]); // Columns
   int nthreads = atoi(argv[3]);

   auto matrix = GenerateRandomMatrix<int>(n, m);
   auto original_matrix = GenerateRandomMatrix<int>(n, m);
   copy_matrix(*original_matrix, *matrix, n*m);
   cout << "Matrix before sequential transpose:" << endl;
   print_matrix(*matrix, m, n);
   sequentialTranspose(*matrix, m, n);
   cout << "Matrix after sequential transpose:" << endl;
   print_matrix(*matrix, n, m);

   // Restore the matrix so that we can test the threaded version.
   copy_matrix(*matrix, *original_matrix, n*m);

   threadedTranspose(*matrix, m, n, nthreads);
   cout << "Matrix after threaded transpose:" << endl;
   print_matrix(*matrix, n, m);
}