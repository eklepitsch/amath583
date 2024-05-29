Name: Evan Klepitsch
Netid: evanklep
Course: AMATH 583
Homework 6

This is my submission for Homework 6.

My code is built using build.sh.

This script will build several executables, located in ./build:
   - xhw6_p1            (Problem 1)

Use the following commands to build and run the code:

chmod +x build.sh
./build.sh
./build/xhw6_p1 <max_problem_size>

The codes above write their results to ./artifacts.

The files associated with each problem are as follows:

Problem 1
---------
./src/p1.cpp
./scripts/p1.slurm
./plots/p1.png

Here is the raw data plotted in p1.png:

problem_size, residual,        log(residual), normalized error, log(normalized error)
16,           4.585973724e-16, -15.33856844,  0.3334182626,     -0.4770106158
32,           8.348112286e-16, -15.07841172,  0.3957054006,     -0.4026280224
64,           1.933226152e-15, -14.71371734,  0.6988718635,     -0.1556024438
128,          2.710985005e-15, -14.56687288,  0.5773519116,     -0.2385593919
256,          3.590414763e-15, -14.44485538,  0.4910805977,     -0.3088472243
512,          5.470082112e-15, -14.26200615,  0.5736292707,     -0.2413686959
1024,         9.718953438e-15, -14.0123805,   0.2733773792,     -0.5632374243
2048,         1.661933513e-14, -13.77938635,  0.480386963,      -0.3184087872
4096,         2.994502754e-14, -13.52367528,  0.8520818366,     -0.06951869225
8192,         5.682794302e-14, -13.24543806,  0.9034847271,     -0.04407918453

Problem 2
---------

Problem 3
---------

Problem 4
---------