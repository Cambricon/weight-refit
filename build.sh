#!/bin/bash
set -e

# build host
mkdir -p build
rm -rf build/*
cd build
cmake ..
make -j4
cd ..
