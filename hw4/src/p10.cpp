#include "strassen.cpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

vector<vector<double>> GenerateSquareMatrix(int n)
{
   vector<vector<double>> matrix(n, vector<double>(n, 0));
   for (int i=0; i<n; i++)
   {
      for (int j=0; j<n; j++)
      {
         matrix[i][j] = (double)(rand() % 1000);
      }
   }
   return matrix;
}

void problem_10()
{
   std::ofstream results;
   results.open("./artifacts/strassen.csv");

   int ntrials = 5;

   int n = 2;
   while(n <= 512)
   {
      long double elapsedtime = 0.L;
      long double avgtime;

      for(unsigned k=0; k<ntrials; ++k)
      {
         auto A = GenerateSquareMatrix(n);
         auto B = GenerateSquareMatrix(n);
         auto start = std::chrono::high_resolution_clock::now();
         auto C = strassenMultiply(A, B);
         auto stop = std::chrono::high_resolution_clock::now();
         auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
         elapsedtime += (duration.count()*1.e-9); //Convert duration to seconds
      }
      avgtime = elapsedtime/static_cast<long double>(ntrials);

      double flops = 3 * n * n;  // 2 multiplies + 1 add for each element
      results << std::setprecision(0) << n << ", " << std::setprecision(10)
         << avgtime << ", " << flops / avgtime << std::endl;

      n <<= 1;
   }

   results.close();
}

int main()
{
   problem_10();
   return 0;
}