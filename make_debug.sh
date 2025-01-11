#!/bin/bash

mkdir -p build

cd build

cmake -D CMAKE_BUILD_TYPE="Debug" -D CMAKE_CXX_FLAGS="-fmax-errors=1" ..

make -j16 $@

