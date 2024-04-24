#!/bin/bash

set -x

INCLUDE_DIR=./include
LIB_DIR=./lib

mkdir -p $LIB_DIR

# Build all the code into one executable
g++ -I $INCLUDE_DIR hw3.cpp mm-jki.cpp mm-kij.cpp ref_daxpy.cpp \
  ref_dgemv.cpp ref_dgemm.cpp -o hw3

# Build the shared object
g++ -c -fPIC -I $INCLUDE_DIR ref_daxpy.cpp -o ref_daxpy.o
g++ -c -fPIC -I $INCLUDE_DIR ref_dgemv.cpp -o ref_dgemv.o
g++ -c -fPIC -I $INCLUDE_DIR ref_dgemm.cpp -o ref_dgemm.o
g++ -c -fPIC -I $INCLUDE_DIR ref_axpyt.cpp -o ref_axpyt.o
g++ -c -fPIC -I $INCLUDE_DIR ref_gemvt.cpp -o ref_gemvt.o
g++ -c -fPIC -I $INCLUDE_DIR ref_gemmt.cpp -o ref_gemmt.o
g++ -shared -o $LIB_DIR/librefBLAS.so ref_daxpy.o ref_dgemv.o \
  ref_dgemm.o ref_axpyt.o ref_gemvt.o ref_gemmt.o

# Build an executable which links with the shared object
g++ hw3.cpp -I $INCLUDE_DIR -DUSE_LIBRARY -L$LIB_DIR -lrefBLAS -o hw3_usesLib
