#include <mpi.h>

template<typename T>
void my_broadcast(T* data, int count, int root, MPI_Comm comm)
{
   // Note: I assume that MPI_Initialize and MPI_Finalize are invoked by the
   // caller (since the caller needs to know the group size to specify root).
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   int count_as_bytes = count * sizeof(T);

   if(rank == root)
   {
      // Sender process
      for(auto i = 0; i<size; ++i)
      {
         // Don't send to self
         if(i != rank)
         {
            // Cast the data to a byte array before sending
            char* data_as_bytes = reinterpret_cast<char*>(data);

            MPI_Request req;
            MPI_Isend(data_as_bytes, count_as_bytes, MPI_UNSIGNED_CHAR,
                      i /*dest*/, 0 /*tag*/, comm, &req);

            std::cout << "Process " << rank << " sent " << count_as_bytes
               << " bytes to process " << i << std::endl;

            // Wait for completion
            MPI_Status status;
            MPI_Wait(&req, &status);
         }
      }
   }
   else
   {
      // Receiver process
      // Allocate a space to receive the data
      char* recv_buf = new char[count_as_bytes];
      MPI_Request req;
      MPI_Irecv(recv_buf, count_as_bytes, MPI_UNSIGNED_CHAR,
                root /*src*/, 0 /*tag*/, comm, &req);

      // Wait for completion
      MPI_Status status;
      MPI_Wait(&req, &status);

      std::cout << "Process " << rank << " received bytes" << std::endl;

      delete[] recv_buf;
   }
}
