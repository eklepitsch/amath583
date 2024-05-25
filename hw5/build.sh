#!/bin/bash

INCLUDE_DIR=$(pwd)/include
SRC_DIR=$(pwd)/src
BUILD_DIR=$(pwd)/build
ARTIFACT_DIR=$(pwd)/artifacts
VENV_DIR=$(pwd)/.venv

mkdir -p $BUILD_DIR
mkdir -p $ARTIFACT_DIR

# Create a Python environment to run Prof. Roche's example code
if [ ! -d "$VENV_DIR" ]; then
  echo "Python environment does not exist. Creating..."
  python -m venv $VENV_DIR
  source $VENV_DIR/bin/activate
  pip install numpy matplotlib PyQt5
fi

set -x

# Problem 1
g++ -g -O0 -I$INCLUDE_DIR -o $BUILD_DIR/xhw5_p1 \
   $SRC_DIR/threaded_life.cpp $SRC_DIR/test-threaded_life.cpp -lpthread
g++ -g -O0 -I$INCLUDE_DIR -o $BUILD_DIR/xhw5_example_p1 \
   $SRC_DIR/conway-life.cpp -lpthread
g++ -g -O0 -I$INCLUDE_DIR -o $BUILD_DIR/xhw5_compare_grids \
   $SRC_DIR/conway-life-compare-grids.cpp

# Problem 2
g++ -g -O0 -std=c++14 -o $BUILD_DIR/xelevator -I$INCLUDE_DIR \
   $SRC_DIR/hw5-elevator.cpp -lpthread

# Problem 3
g++ -g -I$INCLUDE_DIR -o $BUILD_DIR/xhw5_p3 $SRC_DIR/p3.cpp -lopenblas

# Problem 4
nvcc -c -O3 -I$INCLUDE_DIR -arch sm_50 -o $BUILD_DIR/p4.o $SRC_DIR/p4.cpp
g++ -g -o $BUILD_DIR/xhw5_p4 $BUILD_DIR/p4.o -lcudart -lopenblas -lcublas -lm

# Problem 5
g++ -g -o $BUILD_DIR/xhw5_p5 -I$INCLUDE_DIR $SRC_DIR/p5.cpp -lpthread