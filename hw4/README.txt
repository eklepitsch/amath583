Name: Evan Klepitsch
Netid: evanklep
Course: AMATH 583
Homework 4

This is my submission for Homework 4.

My code is built using build.sh.  Note: requires CMake.

This script will build several executables, located in ./build/bin:
   - matrix_class_test     (test cases for Problem 2)
   - hw4_p4                (Problem 4)
   - hw4_p5                (Problem 5)
   - file_swaps_test       (test cases for Problem 5)
   - hw4_p6                (Problem 6)
   - hw4_p7                (Problem 7; run with mpirun)
   - hw4_p9                (Problem 9; run with mpirun)
   - hw4_p10               (Problem 10)
   - strassen_test         (test cases for Problem 10)

Use the following commands to build and run the code:

chmod +x build.sh
./build.sh
./build/bin/hw4_p4
./build/bin/hw4_p5
./build/bin/hw4_p6
# hw4_p7 takes arguments to specify whether to run part a or part b.
mpirun -np <num_procs> ./build/bin/hw4_p7 a                         # scaling efficiency
mpirun -np <num_procs> ./build/bin/hw4_p7 b <num_partition_points>  # numerical error
mpirun -np <num_procs> ./build/bin/hw4_p9 a <num_bytes>             # my_broadcast()
mpirun -np <num_procs> ./build/bin/hw4_p9 b <num_bytes>             # MPI_Broadcast()
./build/bin/hw4_p10

The codes above write their results to ./artifacts.

The slurm files that I used for submitting hw4_p7 and hw4_p9 to HYAK are
located in ./scripts.

The files associated with each problem are as follows:

Problem 2
---------
./include/matrix_class.hpp
./src/matrix_class_test.cpp

Problem 4
---------
./src/p4.cpp
./plots/p4.png

Problem 5
---------
./include/file_swaps.hpp
./src/file_swaps_test.cpp
./src/p5.cpp
./plots/p5.png

Problem 6
---------
./src/p6.cpp
./plots/p6-scaling-efficiency.png
./plots/p6-error.png

Problem 7
---------
./src/p7.cpp
./scripts/p7.slurm
./plots/p7-scaling-efficiency.png
./plots/p7-error.png

Problem 9
---------
./include/my_broadcast.hpp
./src/p9.cpp
./scripts/p9.slurm
./plots/p9.png

Problem 10
----------
./src/strassen.cpp
./src/strassen_test.cpp
./src/p10.cpp
./plots/p10.png

My handwritten solutions for Problems 1, 3, 8 are in the file
evanklep_amath583_hw4.pdf.
