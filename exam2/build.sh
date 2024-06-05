#!/bin/bash

SRC_DIR=$(pwd)/src
BUILD_DIR=$(pwd)/build

mkdir -p $BUILD_DIR

set -x

# Problem 1
g++ -g -std=c++14 -o $BUILD_DIR/xexam2_p1 -I$SRC_DIR $SRC_DIR/mysparse.cpp

# Problem 3
g++ -g -std=c++14 -o $BUILD_DIR/xexam2_p3 -I$SRC_DIR $SRC_DIR/match_pattern.cpp