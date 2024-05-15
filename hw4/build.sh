#!/bin/bash

TOP=$(pwd)
BUILD_DIR=$(pwd)/build
SRC_DIR=$(pwd)/src
INCLUDE_DIR=$(pwd)/include

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake ..
make
