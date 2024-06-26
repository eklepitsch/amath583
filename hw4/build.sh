#!/bin/bash

TOP=$(pwd)
BUILD_DIR=$(pwd)/build
SRC_DIR=$(pwd)/src
INCLUDE_DIR=$(pwd)/include

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake -DCMAKE_BUILD_TYPE=Debug ..
make

mpic++ -o $BUILD_DIR/bin/hw4_p7 -I$INCLUDE_DIR $SRC_DIR/p7.cpp
mpic++ -o $BUILD_DIR/bin/hw4_p9 -I$INCLUDE_DIR $SRC_DIR/p9.cpp
