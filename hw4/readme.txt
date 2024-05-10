Install OpenMPI:

sudo apt install openmpi-bin libopenmpi-dev

Build with mpi:

mpic++ -o hw4_mpi ../hw4_mpi.cpp

Run with mpi:

mpirun -np 8 ./hw4_mpi

