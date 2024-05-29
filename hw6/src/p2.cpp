#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
   if(argc != 2)
   {
      std::cerr << "Usage: " << argv[0] << " <max_copy_size>" << std::endl;
      return 1;
   }
   unsigned long max_copy_size = std::stoul(argv[1]);

   std::ofstream results("./artifacts/p2-results.csv");
   results << "Size (Bytes), H2D Time (ms), H2D Bandwidth (MB/s), "
              "D2H Time (ms), D2H Bandwidth (MB/s)" << std::endl;

   for(unsigned long nbytes=8; nbytes<=max_copy_size; nbytes*=2)
   {
      unsigned ntrials = 100;
      float h2d_avgtime = 0.0;
      float d2h_avgtime = 0.0;
      for(unsigned i=0; i<ntrials; i++)
      {
         uint8_t* hBuf = new uint8_t[nbytes];
         uint8_t* dBuf;

         // Randomly initialize host buffer
         for(unsigned j=0; j<nbytes; j++)
         {
            hBuf[j] = rand() % 256;
         }

         cudaMalloc((void**)&dBuf, nbytes);

         auto start = std::chrono::high_resolution_clock::now();
         cudaMemcpy(dBuf, hBuf, nbytes, cudaMemcpyHostToDevice);
         auto stop = std::chrono::high_resolution_clock::now();
         h2d_avgtime += std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-9; // convert ns to seconds

         start = std::chrono::high_resolution_clock::now();
         cudaMemcpy(hBuf, dBuf, nbytes, cudaMemcpyDeviceToHost);
         stop = std::chrono::high_resolution_clock::now();
         d2h_avgtime += std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-9; // convert ns to seconds

         delete[] hBuf;
         cudaFree(dBuf);
      }
      h2d_avgtime /= ntrials;
      d2h_avgtime /= ntrials;

      results << nbytes << ", " << std::setprecision(10)
         << h2d_avgtime * 1.e3 << ", "
         << (float)nbytes/(h2d_avgtime * 1.e6) << ", "
         << d2h_avgtime * 1.e3 << ", "
         << (float)nbytes/(d2h_avgtime * 1.e6)
         << std::endl;
   }
   return 0;
}