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
# g++ -g -std=c++17 -o $BUILD_DIR/xhw6_p1 $SRC_DIR/p1.cpp -llapacke -lblas

# Problem 2
nvcc -c -O3 -I$INCLUDE_DIR -arch sm_50 -o $BUILD_DIR/p2.o $SRC_DIR/p2.cpp
g++ -g -o $BUILD_DIR/xhw6_p2 $BUILD_DIR/p2.o -lcudart -lm
