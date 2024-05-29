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
   results << "Size (Bytes), H2D Time (ms). H2D Bandwidth (MB/s), "
              "D2H Time (ms), D2H Bandwidth (MB/s)" << std::endl;

   for(unsigned long nbytes=8; nbytes<=max_copy_size; nbytes*=2)
   {
      unsigned ntrials = 3;
      float h2d_avgtime = 0.0;
      float d2h_avgtime = 0.0;
      unsigned long size = nbytes * sizeof(uint8_t);
      for(unsigned i=0; i<ntrials; i++)
      {
         float h2d_time = 0.0;
         float d2h_time = 0.0;
         uint8_t* hBuf = (uint8_t*)malloc(size);
         uint8_t* dBuf;
         cudaMalloc((void**)&dBuf, size);

         // cudaEvent_t start, stop;
         // cudaEventCreate(&start);
         // cudaEventCreate(&stop);

         auto start = std::chrono::high_resolution_clock::now();
         // cudaEventRecord(start);
         cudaMemcpy(dBuf, hBuf, size, cudaMemcpyHostToDevice);
         auto stop = std::chrono::high_resolution_clock::now();
         h2d_avgtime += std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-6;
         // cudaEventRecord(stop);
         // cudaEventSynchronize(stop);
         // cudaEventElapsedTime(&h2d_time, start, stop);

         // h2d_avgtime += h2d_time;

         // cudaEventDestroy(start);
         // cudaEventDestroy(stop);
         // cudaEventCreate(&start);
         // cudaEventCreate(&stop);

         start = std::chrono::high_resolution_clock::now();
         // cudaEventRecord(start);
         cudaMemcpy(hBuf, dBuf, size, cudaMemcpyDeviceToHost);
         stop = std::chrono::high_resolution_clock::now();
         d2h_avgtime += std::chrono::duration_cast<std::chrono::nanoseconds>(
               stop - start).count()*1.e-6;
         // cudaEventRecord(stop);
         // cudaEventSynchronize(stop);
         // cudaEventElapsedTime(&d2h_time, start, stop);
         // d2h_avgtime += d2h_time;

         free(hBuf);
         cudaFree(dBuf);
      }
      h2d_avgtime /= ntrials;
      d2h_avgtime /= ntrials;

      results << size << ", " << std::setprecision(10)
         << h2d_avgtime * 1.e-3 << ", "
         << (float)size/(h2d_avgtime * 1.e3) << ", "
         << d2h_avgtime * (1.e-3) << ", "
         << (float)size/(d2h_avgtime * 1.e3)
         << std::endl;
   }
   return 0;
}