#!/bin/bash

SRC_DIR=$(pwd)/src
BUILD_DIR=$(pwd)/build

mkdir -p $BUILD_DIR

set -x

# Problem 1
g++ -g -std=c++17 -o $BUILD_DIR/xexam2_p1 -I$SRC_DIR $SRC_DIR/mysparse.cpp