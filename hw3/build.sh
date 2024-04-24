#!/bin/bash

set -x

# Build all the code into one executable
g++ hw3.cpp mm-jki.cpp mm-kij.cpp ref_daxpy.cpp ref_dgemv.cpp ref_dgemm.cpp -o hw3

# Build the shared object
g++ -c -fPIC ref_daxpy.cpp -o ref_daxpy.o
g++ -c -fPIC ref_dgemv.cpp -o ref_dgemv.o
g++ -c -fPIC ref_dgemm.cpp -o ref_dgemm.o
g++ -c -fPIC ref_axpyt.cpp -o ref_axpyt.o
g++ -c -fPIC ref_gemvt.cpp -o ref_gemvt.o
g++ -c -fPIC ref_gemmt.cpp -o ref_gemmt.o
g++ -shared -o librefBLAS.so ref_daxpy.o ref_dgemv.o \
  ref_dgemm.o ref_axpyt.o ref_gemvt.o ref_gemmt.o

# Build an executable which links with the shared object
g++ hw3.cpp -DUSE_LIBRARY -L. -lrefBLAS -o hw3_usesLib
