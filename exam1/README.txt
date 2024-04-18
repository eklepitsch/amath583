Name: Evan Klepitsch
Netid: evanklep
Course: AMATH 583
Exam 1

This is my submission for Exam 1.

My code can be built using the script build.sh.

This script will build two executables:
   - xtst-ijk-perms     (Problem 1)
   - xtst-matrix-swaps  (Problem 2)

In summary, run the following commands to build and run the code:

chmod +x build.sh
./build.sh
./xtst-ijk-perms
./xtst-matrix-swaps

My plot for Problem 2 is contained in the file swap-performance.png.  The data
from my machine (which was used to generate the plot) is below:

Dimension   Row swap time (s)   Column swap time (s)
16          2.24E-07            2.22E-07
32          4.20E-07            3.32E-07
64          6.83E-07            6.10E-07
128         2.47E-06            1.19E-06
256         7.45E-06            2.34E-06
512         2.83E-05            4.86E-06
1024        6.03E-05            9.60E-06
2048        0.000110511         1.98E-05
4096        0.00023776          3.88E-05

My handwritten solutions for Problems 3-5 are in the file evanklep_exam1.pdf.
