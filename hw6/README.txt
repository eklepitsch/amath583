Name: Evan Klepitsch
Netid: evanklep
Course: AMATH 583
Homework 6

This is my submission for Homework 6.

My code is built using build.sh.

This script will build several executables, located in ./build:
   - xhw6_p1            (Problem 1)
   - xhw6_p2            (Problem 2)
   - xgrad-fftw         (Problem 3)

Use the following commands to build and run the code:

chmod +x build.sh
./build.sh
./build/xhw6_p1 <max_problem_size>
./build/xhw6_p2 <max_copy_size>
./build/xgrad-fftw

The codes above write their results to ./artifacts.

My working directory on Hyak is /gscratch/amath/evanklep/hw6.

The files associated with each problem are as follows:

Problem 1
---------
./src/p1.cpp
./scripts/p1.slurm
./plots/p1.png

Problem 2
---------
./src/p2.cpp
./scripts/p2.slurm
./plots/p2.png

Problem 3
---------
./src/gradient-fftw.cpp
./plots/p3-normal-scale.png
./plots/p3-log-scale.png

Problem 4
---------
evanklep_hw6.pdf (handwritten solutions)