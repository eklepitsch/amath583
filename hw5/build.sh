#!/bin/bash

INCLUDE_DIR=$(pwd)/include
SRC_DIR=$(pwd)/src
BUILD_DIR=$(pwd)/build
ARTIFACT_DIR=$(pwd)/artifacts

mkdir -p $BUILD_DIR
mkdir -p $ARTIFACT_DIR

g++ -g -I$INCLUDE_DIR -o $BUILD_DIR/xhw5_p3 $SRC_DIR/p3.cpp -lopenblas
nvcc -g -I$INCLUDE_DIR -o $BUILD_DIR/xhw5_p4 $SRC_DIR/p4.cpp \
   -lopenblas -lcublas -lcudart