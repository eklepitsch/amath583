Name: Evan Klepitsch
Netid: evanklep
Course: AMATH 583
Homework 4

This is my submission for Homework 4.

My code can be built using the script build.sh.  Note: requires CMake.

This script will build several executables, located in ./build/bin:
   - matrix_class_test     (test cases for Problem 2)
   - hw4_p4                (Problem 4)
   - hw4_p5                (Problem 5)
   - file_swaps_test       (test cases for Problem 5)
   - hw4_p6                (Problem 6)
   - hw4_p7                (Problem 7; run with mpirun)

In summary, run the following commands to build and run the code:

chmod +x build.sh
./build.sh
./build/bin/hw4_p4
./build/bin/hw4_p5
./build/bin/hw4_p6
# hw4_p7 takes arguments to specify whether to run part a or part b.
mpirun -np <num_procs> ./build/bin/hw4_p7 a
mpirun -np <num_procs> ./build/bin/hw4_p7 b <num_partition_points>

The codes above write their results to ./artifacts.

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
./plots/p7-scaling-efficiency.png
./plots/p7-error.png

Problem 9
---------
To do.

Problem 10
----------
To do.

My handwritten solutions for Problems 1, 3, 8 are in the file
evanklep_amath583_hw4.pdf.
