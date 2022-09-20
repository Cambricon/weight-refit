#!/bin/bash
set -e

# build host
mkdir -p build
rm -rf build/*
cd build
cmake ..
make
cd ..


mkdir -p build
rm -rf build/*
cd build
cmake ..
make
cd ..