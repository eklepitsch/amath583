#!/bin/bash

INCLUDE_DIR=$(pwd)/include
SRC_DIR=$(pwd)/src
BUILD_DIR=$(pwd)/build
ARTIFACT_DIR=$(pwd)/artifacts
VENV_DIR=$(pwd)/.venv

mkdir -p $BUILD_DIR
mkdir -p $ARTIFACT_DIR

set -x

# Problem 1
g++ -g -std=c++17 -o $BUILD_DIR/xhw6_p1 $SRC_DIR/p1.cpp -llapacke -lblas

# Problem 2
nvcc -c -O3 -I$INCLUDE_DIR -arch sm_50 -o $BUILD_DIR/p2.o $SRC_DIR/p2.cpp
g++ -g -o $BUILD_DIR/xhw6_p2 $BUILD_DIR/p2.o -lcudart -lm

# Problem 3
cp $SRC_DIR/gradient-fftw.cpp $SRC_DIR/gradient-fftw.cu
nvcc -c -arch sm_50 -std=c++14 -I$INCLUDE_DIR -I/usr/local/include \
   -I/usr/local/cuda-12.1/include $SRC_DIR/gradient-fftw.cpp
nvcc -o $BUILD_DIR/xgrad-fftw gradient-fftw.o -L/usr/local/lib -lfftw3 \
   -L/usr/local/cuda-12.1/lib64 -lcudart -lcufft -lm
