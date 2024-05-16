#include "my_broadcast.hpp"
#include <cstdlib>
#include <cstring>


template<typename T>
void mpi_broadcast(T* data, int count, int root, MPI_Comm comm)
{
   return;
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

   MPI_Finalize();

   delete[] data;
   return 0;
}