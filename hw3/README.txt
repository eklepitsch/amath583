#!/bin/bash

INCLUDE_DIR=./include
LIB_DIR=./lib
mkdir -p $LIB_DIR
g++ -c -fPIC -I $INCLUDE_DIR ref_daxpy.cpp -o ref_daxpy.o
g++ -c -fPIC -I $INCLUDE_DIR ref_dgemv.cpp -o ref_dgemv.o
g++ -c -fPIC -I $INCLUDE_DIR ref_dgemm.cpp -o ref_dgemm.o
g++ -c -fPIC -I $INCLUDE_DIR ref_axpyt.cpp -o ref_axpyt.o
g++ -c -fPIC -I $INCLUDE_DIR ref_gemvt.cpp -o ref_gemvt.o
g++ -c -fPIC -I $INCLUDE_DIR ref_gemmt.cpp -o ref_gemmt.o
g++ -shared -o $LIB_DIR/librefBLAS.so ref_daxpy.o ref_dgemv.o \
  ref_dgemm.o ref_axpyt.o ref_gemvt.o ref_gemmt.o
