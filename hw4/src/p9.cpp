#include "my_broadcast.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>

template<typename T>
void mpi_broadcast(T* data, int count, int root, MPI_Comm comm)
{
   // Note: I assume that MPI_Initialize and MPI_Finalize are invoked by the
   // caller (since the caller needs to know the group size to specify root).
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   int count_as_bytes = count * sizeof(T);

   // Cast the data to a byte array before sending
   char* data_as_bytes = reinterpret_cast<char*>(data);

   MPI_Bcast(data_as_bytes, count_as_bytes, MPI_UNSIGNED_CHAR,
             root /*root*/, comm);

   if(rank == root)
   {
      std::cout << "Process " << rank << " sent " << count_as_bytes
         << " bytes as broadcast" << std::endl;
   }
   else
   {
      std::cout << "Process " << rank << " received broadcast." << std::endl;
   }
}

int main(int argc, char** argv)
{
   // First argument specifies part a or part b (homebrew vs MPI bcast).
   // Second argument is the number of bytes to send.
   if(argc != 3)
   {
      throw std::invalid_argument("Must provide two arguments");
   }

   int nbytes = std::stoi(std::string(argv[2]));
   if(nbytes <= 0 || nbytes % sizeof(int))
   {
      throw std::invalid_argument("Invalid data size");
   }

   // Randomly initialize the data
   int int_array_size = nbytes/sizeof(int);
   int* data = new int[int_array_size];
   for(auto i=0; i<int_array_size; ++i)
   {
      data[i] = std::rand() % 1000;
   }

   MPI_Init(nullptr, nullptr);

   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   // Randomize the sender
   int sender = std::rand() % size;

   MPI_Barrier(MPI_COMM_WORLD);
   auto start = std::chrono::high_resolution_clock::now();

   if(!std::strcmp(argv[1], "a"))
   {
      // Homebrew broadcast
      my_broadcast(data, int_array_size, sender, MPI_COMM_WORLD);
   }
   else if(!std::strcmp(argv[1], "b"))
   {
      // MPI_Bcast
      mpi_broadcast(data, int_array_size, sender, MPI_COMM_WORLD);
   }
   else
   {
      throw std::invalid_argument("First argument must be a or b");
   }

   MPI_Barrier(MPI_COMM_WORLD);
   auto stop = std::chrono::high_resolution_clock::now();

   MPI_Finalize();

   if(rank == sender)
   {
      // Only the sender will record the elapsed time.
      auto duration =
         std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      long double elapsed_time = duration.count()*1.e-9;  // Convert duration to seconds

      std::ofstream results;
      results.open("./p9-results.csv", std::ios::ate | std::ios::app);
      results << argv[1] << ", " << size << ", " << nbytes << ", "
         << std::setprecision(10) << elapsed_time << ", "
         << (long double)nbytes/elapsed_time << std::endl;
      results.close();
   }

   delete[] data;
   return 0;
}