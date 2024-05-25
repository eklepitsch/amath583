Name: Evan Klepitsch
Netid: evanklep
Course: AMATH 583
Homework 5

This is my submission for Homework 5.

My code is built using build.sh.  Note: requires OpenBLAS to be installed
(uses `-lopenblas` when compiling).

This script will build several executables, located in ./build:
   - xhw5_p1               (Problem 1)
   - xelevator             (Problem 2)
   - xhw5_p3               (Problem 3)
   - xhw5_p4               (Problem 4)

Use the following commands to build and run the code:

chmod +x build.sh
./build.sh
./build/xhw5_p1 <gridSize> <numThreads>
./build/xelevator <numPeople>
./build/xhw5_p3
./build/xhw5_p4 <maxMatrixDimension>

The codes above write their results to ./artifacts.

The files associated with each problem are as follows:

Problem 1
---------
./src/threaded_life.cpp
./src/test-threaded_life.cpp

Problem 2
---------
./include/hw5-elevator.hpp
./src/hw5-elevator.cpp

Problem 3
---------
./src/p3.cpp
./plots/p3-log-scale.png
./plots/p3-normal-scale.png

Problem 4
---------
./src/p4.cpp
./scripts/p4.slurm
./plots/p4-log-scale.png
./plots/p4-normal-scale.png